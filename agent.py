"""Agent Framework agent with Cosmos DB long-term memory."""
from __future__ import annotations

import os

from agent_framework import Agent, AgentSession
from agent_framework.foundry import FoundryChatClient
from azure.ai.agentserver.core import get_request_context
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

from memory_provider import DEFAULT_USER, CosmosMemoryProvider
from memory_store import MemoryStore, get_credential

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

INSTRUCTIONS = "You are a helpful assistant."


class MemoryAgent(Agent):
    """Tags every session with the caller so memory is scoped per user."""

    def __init__(self, *args, memory: CosmosMemoryProvider, **kwargs) -> None:
        super().__init__(*args, context_providers=[memory], **kwargs)
        self._memory = memory

    def create_session(self, *, session_id: str | None = None, user_id: str | None = None) -> AgentSession:
        session = super().create_session(session_id=session_id)
        if user_id is None:
            try:
                user_id = get_request_context().user_id  # x-agent-user-id header set by Foundry
            except Exception:
                user_id = None
        session.state.setdefault(self._memory.source_id, {})["user_id"] = user_id or DEFAULT_USER
        return session


def build_agent() -> MemoryAgent:
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    model = os.getenv("CHAT_MODEL", "gpt-4o")
    credential = get_credential()
    oai = AIProjectClient(endpoint=endpoint, credential=credential).get_openai_client().with_options(
        timeout=60, max_retries=3)
    memory = CosmosMemoryProvider(MemoryStore(), oai, model)
    client = FoundryChatClient(project_endpoint=endpoint, model=model, credential=credential)
    return MemoryAgent(
        client=client,
        instructions=INSTRUCTIONS,
        # The host supplies conversation history, so the service need not store it.
        default_options={"store": False},
        memory=memory,
    )
