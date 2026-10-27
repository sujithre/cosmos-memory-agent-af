"""Foundry hosted-agent entry point (Responses protocol) for the Agent Framework agent."""
from agent_framework.foundry import ResponsesHostServer

from agent import build_agent

if __name__ == "__main__":
    ResponsesHostServer(build_agent()).run()
