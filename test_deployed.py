"""Call the deployed Agent Framework hosted agent through the Foundry endpoint (same path as the playground).

Verifies recall across conversations, then reads Cosmos directly (cross-partition) and removes what the test wrote.
"""
import os
import sys
import time
from datetime import datetime, timezone

from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

from memory_store import MemoryStore, get_credential

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
start = datetime.now(timezone.utc).isoformat()
project = AIProjectClient(endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=get_credential())
oai = project.get_openai_client(agent_name=os.getenv("AGENT_NAME", "cosmos-memory-agent-af")).with_options(
    timeout=180, max_retries=1)

print("turn 1:", oai.responses.create(input="Hi, I'm Sujith, I work on CSOC Engineering and I love sweets. "
                                           "Please keep answers short.").output_text)
r = oai.responses.create(input="What do you remember about me?")  # new conversation
print("turn 2:", r.output_text)

store = MemoryStore()
rows = list(store.memories.query_items("SELECT * FROM c WHERE c.updated_at >= @s",
                                       parameters=[{"name": "@s", "value": start}],
                                       enable_cross_partition_query=True))
print("\nstored by this test:")
for m in rows:
    print(f"  user={m['user_id'][:12]}.. [{m['kind']}] {m['content']}")
if "--keep" not in sys.argv:
    for m in rows:
        store.memories.delete_item(m["id"], partition_key=m["user_id"])
    print(f"cleaned {len(rows)}")
