"""Latency breakdown: where does a turn spend its time?"""
import asyncio
import sys
import time
import uuid

from agent import build_agent

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def t(label, fn, *a):
    s = time.perf_counter()
    r = fn(*a)
    print(f"  {label:<34}{(time.perf_counter() - s) * 1000:7.0f} ms")
    return r


async def main():
    agent = build_agent()
    p = agent._memory
    store, user, sid = p.store, f"lat-{uuid.uuid4().hex[:6]}", "s1"
    store.add(user, "user_preference", "Likes sweets.")
    print("components:")
    t("recall_block (1 Cosmos query)", store.recall_block, user)
    t("add_turn x2 (2 Cosmos writes)", lambda: (store.add_turn(user, sid, "user", "hi"), store.add_turn(user, sid, "assistant", "hello")))
    t("extract (reads + model + writes)", p._extract, user, sid)
    s = time.perf_counter()
    p.oai.responses.create(model=p.model, input="Say hi")
    print(f"  {'bare model call (no memory)':<34}{(time.perf_counter() - s) * 1000:7.0f} ms")
    print("end to end agent.run():")
    sess = agent.create_session(user_id=user)
    for i in range(2):
        s = time.perf_counter()
        await agent.run("What do I like?", session=sess)
        print(f"  run #{i + 1:<31}{(time.perf_counter() - s) * 1000:7.0f} ms (reply returned)")
        await p.drain()
    store.delete_all(user)


asyncio.run(main())
