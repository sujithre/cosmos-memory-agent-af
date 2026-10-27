"""Local test: write in session 1, recall in a brand-new session 2, then additive + replacement facts."""
import asyncio
import sys
import uuid

from agent import build_agent

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def say(agent, user, session, msg):
    r = await agent.run(msg, session=session)
    print(f"you  : {msg}\nagent: {r.text}")
    await agent._memory.drain()  # background extraction finishes before the next step


def dump(store, user):
    for m in store.list(user):
        print(f"   [{m['updated_at'][:10]}] {m['kind']:<15} {m['content']}")


async def main():
    agent = build_agent()
    store = agent._memory.store
    user = f"af-test-{uuid.uuid4().hex[:6]}"
    try:
        print("== session 1 ==")
        s1 = agent.create_session(user_id=user)
        await say(agent, user, s1, "Hi, I'm Sujith, I work on CSOC engineering and I love sweets. Keep answers short.")
        dump(store, user)
        print("\n== session 2 (new conversation) ==")
        s2 = agent.create_session(user_id=user)
        await say(agent, user, s2, "What do you remember about me?")
        print("\n== additive / replacement ==")
        await say(agent, user, s2, "I love chocolates too.")
        await say(agent, user, s2, "Actually my favourite sweet is now chocolate.")
        dump(store, user)
        s3 = agent.create_session(user_id=user)
        await say(agent, user, s3, "What's my favourite sweet?")
        print("\n== explicit tool: remember / forget ==")
        await say(agent, user, s3, "Remember that my team is called Blue Squad.")
        dump(store, user)
    finally:
        print("\ncleanup:", store.delete_all(user))


asyncio.run(main())
