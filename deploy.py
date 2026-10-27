"""Package the agent source and create a Foundry hosted-agent version (source-code deployment, no Docker).

Usage: python deploy.py
Reads FOUNDRY_PROJECT_ENDPOINT, CHAT_MODEL, COSMOS_ENDPOINT, COSMOS_DATABASE from .env.
"""
import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeConfiguration, HostedAgentDefinition, ProtocolVersionRecord
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
AGENT_NAME = os.getenv("AGENT_NAME", "cosmos-memory-agent-af")
FILES = ["main.py", "agent.py", "memory_provider.py", "memory_store.py", "requirements.txt"]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(HERE / f, f)
zip_bytes = buf.getvalue()
sha = hashlib.sha256(zip_bytes).hexdigest()
print(f"zip: {len(zip_bytes)} bytes, sha256={sha[:12]}...")

project = AIProjectClient(endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                          credential=AzureCliCredential(process_timeout=90))
created = project.agents.create_version_from_code(
    agent_name=AGENT_NAME,
    definition=HostedAgentDefinition(
        cpu="1", memory="2Gi",
        code_configuration=CodeConfiguration(
            runtime="python_3_13", entry_point=["python", "main.py"], dependency_resolution="remote_build"),
        protocol_versions=[ProtocolVersionRecord(protocol="responses", version="2.0.0")],
        environment_variables={
            "CHAT_MODEL": os.getenv("CHAT_MODEL", "gpt-4o"),
            "COSMOS_ENDPOINT": os.environ["COSMOS_ENDPOINT"],
            "COSMOS_DATABASE": os.getenv("COSMOS_DATABASE", "custom_memory"),
        },
    ),
    code=("agent-code.zip", zip_bytes, "application/zip"),
    code_zip_sha256=sha,
    description="Agent Framework agent with Cosmos DB long-term memory (profile, preferences, chat summary)",
)
print(f"created {AGENT_NAME} version {created.version}")
try:
    print("instance_identity:", created["instance_identity"])
except Exception:
    pass

while True:
    v = project.agents.get_version(agent_name=AGENT_NAME, agent_version=created.version)
    status = v["status"]
    print("status:", status)
    if status == "active":
        print("instance_identity:", v.get("instance_identity"))
        break
    if status == "failed":
        print("error:", v.get("error"))
        sys.exit(1)
    time.sleep(10)
print("ready")
