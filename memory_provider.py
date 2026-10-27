"""Agent Framework context provider backed by Cosmos DB (user_profile, user_preference, chat_summary).

before_run : inject the user's memories (newest first, dated) into the instructions and expose the
             memory_remember / memory_forget tools bound to the caller.
after_run  : save the turn, then ask the model to update profile/preferences and the rolling summary.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Annotated, Any, Literal

from agent_framework import AgentSession, ContextProvider, SessionContext, tool
from pydantic import Field

from memory_store import KINDS, MemoryStore

MEMORY_INSTRUCTIONS = (
    "You have long-term memory. Use the MEMORY section to personalise answers naturally; never mention the "
    "memory mechanism unless asked. Memories carry the date they were last updated: if two memories conflict "
    "(for example an old and a new favourite, location or role), trust the most recent one, and when asked for "
    "'my favourite X' prefer the most recently stated. If the user explicitly asks you to remember something "
    "call memory_remember; if they ask you to forget something call memory_forget (ids are shown as [id] in "
    "MEMORY). Do not store secrets or passwords."
)

EXTRACT_PROMPT = """You maintain long-term memory for a chat assistant. Return JSON only.

Existing memories (id | kind | last updated | content):
{existing}

Previous conversation summary (may be empty):
{summary}

Latest conversation turns:
{turns}

Return a JSON object:
{{
  "ops": [
    {{"op": "add", "kind": "user_profile|user_preference", "content": "..."}},
    {{"op": "update", "id": "<existing id>", "content": "..."}},
    {{"op": "delete", "id": "<existing id>"}}
  ],
  "summary": "2-4 sentence rolling summary of this whole conversation so far"
}}
Rules:
- user_profile = durable facts about the user (name, role, team, location, projects).
- user_preference = how the user likes things (tone, format, likes, dislikes).
- Only store what the USER stated or clearly implied. Never store secrets, passwords or one-off task details.
- Each memory is ONE atomic fact. Do not combine facts, and do not add anything already covered by an existing memory. "Likes X", "Loves X" and "Enjoys X" are the SAME fact: never add a second one.
- Additive facts are kept side by side (e.g. "Likes sweets." and "Likes chocolates.").
- If the user states something that REPLACES or CONTRADICTS an existing memory about the same single-valued attribute (favourite X, current role, location, team, name, "now"/"anymore"/"actually"/"changed"), use update on that memory (or delete it) instead of adding a second one.
- Prefer update over add when a memory is changed; delete when the user retracts something.
- Short third-person sentences ("Likes sweets."). Use an empty ops list if nothing new."""

DEFAULT_USER = "anonymous"


class CosmosMemoryProvider(ContextProvider):
    def __init__(self, store: MemoryStore, oai, model: str, source_id: str = "cosmos_memory") -> None:
        super().__init__(source_id)
        self.store = store
        self.oai = oai
        self.model = model
        self._extract_lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()

    def _user_id(self, state: dict[str, Any]) -> str:
        return state.get("user_id") or DEFAULT_USER

    # ---- tools -------------------------------------------------------------
    def _tools(self, user_id: str) -> list:
        store = self.store

        @tool(
            name="memory_remember",
            description="Save a durable fact about the user. kind=user_profile for who the user is (name, role, "
                        "team, location, projects); kind=user_preference for how they like things (tone, format, "
                        "likes/dislikes).",
        )
        async def memory_remember(
            kind: Annotated[Literal["user_profile", "user_preference"], Field(description="Memory kind")],
            content: Annotated[str, Field(description="One short third-person sentence, e.g. 'Likes sweets.'")],
        ) -> str:
            m = await asyncio.to_thread(store.add, user_id, kind, content, "tool")
            return json.dumps({"saved": m["id"]})

        @tool(name="memory_forget", description="Delete a saved memory by its id (shown as [id] in MEMORY).")
        async def memory_forget(memory_id: Annotated[str, Field(description="Memory id")]) -> str:
            return json.dumps({"deleted": await asyncio.to_thread(store.delete, user_id, memory_id)})

        return [memory_remember, memory_forget]

    # ---- recall ------------------------------------------------------------
    async def before_run(self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]) -> None:
        user_id = self._user_id(state)
        block = await asyncio.to_thread(self.store.recall_block, user_id)
        text = MEMORY_INSTRUCTIONS + (f"\n\nMEMORY\n{block}" if block else "")
        context.extend_instructions(self.source_id, text)
        context.extend_tools(self.source_id, self._tools(user_id))

    # ---- save --------------------------------------------------------------
    async def after_run(self, *, agent, session: AgentSession, context: SessionContext, state: dict[str, Any]) -> None:
        """Returns immediately; saving and extraction run in the background so replies are not delayed."""
        user_id = self._user_id(state)
        session_id = session.session_id or context.session_id or "session"
        user_text = next((m.text for m in reversed(context.input_messages) if m.role == "user" and m.text), "")
        reply = context.response.text if context.response else ""
        if not user_text:
            return
        task = asyncio.create_task(asyncio.to_thread(self._save_and_extract, user_id, session_id, user_text, reply))
        self._tasks.add(task)  # keep a reference so the task is not garbage collected
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for pending background saves (tests / graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def _save_and_extract(self, user_id: str, session_id: str, user_text: str, reply: str) -> None:
        try:
            self.store.add_turn(user_id, session_id, "user", user_text)
            if reply:
                self.store.add_turn(user_id, session_id, "assistant", reply)
            with self._extract_lock:
                self._extract(user_id, session_id)
        except Exception as e:  # memory must never break the chat
            print(f"[memory save failed: {type(e).__name__}: {e}]", flush=True)

    def _extract(self, user_id: str, session_id: str) -> None:
        summary = self.store.get_summary(user_id, session_id)
        turns = self.store.recent_turns(user_id, session_id)
        existing = [m for m in self.store.list(user_id) if m["kind"] != "chat_summary"]
        prompt = EXTRACT_PROMPT.format(
            existing="\n".join(f"{m['id']} | {m['kind']} | {m['updated_at'][:10]} | {m['content']}"
                               for m in existing) or "(none)",
            summary=summary or "(none)",
            turns="\n".join(f"{t['role']}: {t['text']}" for t in turns),
        )
        resp = self.oai.responses.create(model=self.model, input=prompt, text={"format": {"type": "json_object"}})
        data = json.loads((resp.output_text or "").strip())
        known = {m["id"] for m in existing}
        for op in data.get("ops", []):
            kind = op.get("kind")
            if op.get("op") == "add" and kind in KINDS and kind != "chat_summary":
                self.store.add(user_id, kind, op["content"])
            elif op.get("op") == "update" and op.get("id") in known:
                self.store.update(user_id, op["id"], op["content"])
            elif op.get("op") == "delete" and op.get("id") in known:
                self.store.delete(user_id, op["id"])
        if data.get("summary"):
            self.store.upsert_summary(user_id, session_id, data["summary"])
