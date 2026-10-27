# Cosmos DB memory agent (Microsoft Agent Framework + Foundry hosted agent)

An Agent Framework agent with long-term memory in Azure Cosmos DB, hosted in Microsoft Foundry
(source-code deployment, no Docker). It remembers three kinds of memory per user across conversations:

- `user_profile`: name, role, team, location
- `user_preference`: likes, tone, format
- `chat_summary`: rolling summary per conversation

Memories carry dates, newest first. If two conflict, the most recent wins. Additive facts ("likes sweets",
"likes chocolates") are kept side by side. A statement that replaces one ("my favourite is now X") updates it.

## How it works

| File | Role |
|---|---|
| `memory_provider.py` | Agent Framework `ContextProvider`. `before_run` injects memories and the `memory_remember` / `memory_forget` tools. `after_run` saves the turn and extracts memories in the background, so replies are not delayed. |
| `memory_store.py` | Cosmos CRUD (`memories`, `turns`). Uses Entra auth, no keys. |
| `agent.py` | `MemoryAgent`: scopes each session to the caller (`x-agent-user-id`). |
| `main.py` | `ResponsesHostServer` entry point for the Foundry hosted agent. |
| `deploy.py` | Zips the source and creates a hosted-agent version. |
| `test_local.py`, `test_deployed.py`, `bench.py` | Local test, deployed test, latency breakdown. |

## Cosmos setup

Database `custom_memory` with two containers, both partitioned by `/user_id`:

```powershell
az cosmosdb sql database create -a <cosmos-acct> -g <rg> -n custom_memory
az cosmosdb sql container create -a <cosmos-acct> -g <rg> -d custom_memory -n memories -p /user_id
az cosmosdb sql container create -a <cosmos-acct> -g <rg> -d custom_memory -n turns -p /user_id --ttl 2592000
```

Permissions:

| Who | Role | Scope |
|---|---|---|
| Hosted agent `instance_identity` | Cosmos DB Built-in Data Contributor (`00000000-0000-0000-0000-000000000002`) | database `/dbs/custom_memory` |
| Developers testing locally | same data role | same scope |
| Developers deploying | Foundry User | Foundry project |
| Developers testing locally | Cognitive Services OpenAI User | Foundry account |

```powershell
az cosmosdb sql role assignment create -a <cosmos-acct> -g <rg> `
  --role-definition-id 00000000-0000-0000-0000-000000000002 `
  --principal-id <object-id> --scope "/dbs/custom_memory"
```

The agent's `instance_identity` is printed by `deploy.py`. It exists only after the first deploy, so grant the role then.

## Run

```powershell
python -m venv venv
.\venv\Scripts\python -m pip install --pre -r requirements.txt
copy .env.example .env      # fill in endpoints; run `az login`
python test_local.py        # local test against Cosmos
python deploy.py            # creates a new agent version in Foundry
python test_deployed.py     # calls the deployed agent
```

## Notes

- `agent-framework-foundry-hosting` is a pre-release and is pinned.
- Latency: the first message of each conversation pays the hosted-agent sandbox cold start (about 10-15 s).
  Later turns in the same conversation take about 2 s including memory.
- Run `python test_deployed.py --keep` to keep the rows the test writes.
