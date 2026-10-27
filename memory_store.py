"""Cosmos DB backed long-term memory: user_profile, user_preference, chat_summary."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from azure.core import MatchConditions
from azure.cosmos import CosmosClient, exceptions
from azure.identity import AzureCliCredential, DefaultAzureCredential

KINDS = ("user_profile", "user_preference", "chat_summary")
MAX_ITEMS_PER_KIND = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", "", text.lower())
    t = re.sub(r"\b(likes?|loves?|enjoys?|prefers?|really|also|too)\b", "likes", t)
    return re.sub(r"\s+", " ", re.sub(r"\b(likes )+", "likes ", t)).strip()


def get_credential():
    # Hosted agents get a platform identity (DefaultAzureCredential); locally use the `az login` session.
    if os.getenv("FOUNDRY_AGENT_NAME"):
        return DefaultAzureCredential()
    return AzureCliCredential(process_timeout=90)


class MemoryStore:
    def __init__(self, endpoint: Optional[str] = None, database: Optional[str] = None):
        endpoint = endpoint or os.environ["COSMOS_ENDPOINT"]
        database = database or os.getenv("COSMOS_DATABASE", "custom_memory")
        db = CosmosClient(endpoint, credential=get_credential()).get_database_client(database)
        self.memories = db.get_container_client("memories")
        self.turns = db.get_container_client("turns")

    # ---- memories -------------------------------------------------------
    def list(self, user_id: str, kind: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM c WHERE c.user_id = @u"
        params = [{"name": "@u", "value": user_id}]
        if kind:
            query += " AND c.kind = @k"
            params.append({"name": "@k", "value": kind})
        query += " ORDER BY c.updated_at DESC"
        return list(self.memories.query_items(query, parameters=params, partition_key=user_id))

    def add(self, user_id: str, kind: str, content: str, source: str = "extraction") -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        content = content.strip()
        if not content:
            raise ValueError("content is empty")
        # Skip duplicates within the same kind, treating like/love/enjoy/prefer wording as equivalent.
        for m in self.list(user_id, kind):
            if _norm(m["content"]) == _norm(content):
                return m
        now = _now()
        item = {
            "id": uuid.uuid4().hex[:12], "user_id": user_id, "kind": kind,
            "content": content, "source": source, "created_at": now, "updated_at": now,
        }
        self.memories.create_item(item)
        self._trim(user_id, kind)
        return item

    def update(self, user_id: str, memory_id: str, content: str) -> Optional[dict]:
        try:
            item = self.memories.read_item(memory_id, partition_key=user_id)
        except exceptions.CosmosResourceNotFoundError:
            return None
        item["content"] = content.strip()
        item["updated_at"] = _now()
        return self.memories.replace_item(item, item, etag=item["_etag"],
                                          match_condition=MatchConditions.IfNotModified)

    def delete(self, user_id: str, memory_id: str) -> bool:
        try:
            self.memories.delete_item(memory_id, partition_key=user_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    def delete_all(self, user_id: str) -> int:
        n = 0
        for m in self.list(user_id):
            n += self.delete(user_id, m["id"])
        return n

    def upsert_summary(self, user_id: str, session_id: str, content: str) -> dict:
        """One rolling chat_summary per session (id = session id) so the same chat isn't duplicated."""
        item = {
            "id": f"sum-{session_id}", "user_id": user_id, "kind": "chat_summary",
            "content": content.strip(), "source": "summary", "session_id": session_id,
            "created_at": _now(), "updated_at": _now(),
        }
        try:
            old = self.memories.read_item(item["id"], partition_key=user_id)
            item["created_at"] = old["created_at"]
        except exceptions.CosmosResourceNotFoundError:
            pass
        self.memories.upsert_item(item)
        self._trim(user_id, "chat_summary")
        return item

    def get_summary(self, user_id: str, session_id: str) -> str:
        try:
            return self.memories.read_item(f"sum-{session_id}", partition_key=user_id)["content"]
        except exceptions.CosmosResourceNotFoundError:
            return ""

    def _trim(self, user_id: str, kind: str) -> None:
        items = self.list(user_id, kind)
        for old in items[MAX_ITEMS_PER_KIND:]:
            self.delete(user_id, old["id"])

    # ---- raw turns (30 day TTL set on the container) ------------------------
    def add_turn(self, user_id: str, session_id: str, role: str, text: str) -> None:
        self.turns.create_item({
            "id": uuid.uuid4().hex, "user_id": user_id, "session_id": session_id,
            "role": role, "text": text, "ts": _now(),
        })

    def recent_turns(self, user_id: str, session_id: str, limit: int = 12) -> list[dict]:
        q = ("SELECT TOP @n c.role, c.text, c.ts FROM c WHERE c.user_id=@u AND c.session_id=@s "
             "ORDER BY c.ts DESC")
        rows = list(self.turns.query_items(
            q, parameters=[{"name": "@n", "value": limit}, {"name": "@u", "value": user_id},
                           {"name": "@s", "value": session_id}], partition_key=user_id))
        return list(reversed(rows))

    # ---- recall -------------------------------------------------------------
    def recall_block(self, user_id: str, max_summaries: int = 3) -> str:
        """Text block injected into the model instructions (newest first, with dates and ids)."""
        mem = self.list(user_id)
        by_kind = {k: [m for m in mem if m["kind"] == k] for k in KINDS}
        lines = []
        if by_kind["user_profile"]:
            lines.append("User profile (newest first):")
            lines += [f"- [{m['id']}] ({m['updated_at'][:10]}) {m['content']}" for m in by_kind["user_profile"]]
        if by_kind["user_preference"]:
            lines.append("User preferences (newest first):")
            lines += [f"- [{m['id']}] ({m['updated_at'][:10]}) {m['content']}" for m in by_kind["user_preference"]]
        if by_kind["chat_summary"]:
            lines.append("Recent conversation summaries:")
            lines += [f"- ({m['updated_at'][:10]}) {m['content']}"
                      for m in by_kind["chat_summary"][:max_summaries]]
        return "\n".join(lines)
