import asyncio
import time

from app.config import RuntimeConfig, Settings
from app.engine import QueueFull, RequestTimeout, RequestTooLarge, SimEngine

FAST = Settings(sim_speed=20.0)
PROMPT = " ".join(f"w{i}" for i in range(100))   # 100 tokens => 6 full 16-token blocks


def run(coro):
    return asyncio.run(coro)


async def started(settings=FAST, **cfg) -> SimEngine:
    engine = SimEngine(settings, RuntimeConfig(**cfg))
    await engine.start()
    return engine


def test_generates_exactly_max_tokens_deterministically():
    async def go():
        e = await started()
        a = await e.generate("t", "hello world", 10)
        b = await e.generate("t", "hello world", 10)
        await e.stop()
        return a, b

    a, b = run(go())
    assert a.completion_tokens == 10 and len(a.text.split()) == 10
    assert a.tokens == b.tokens
    assert a.ttft_ms is not None and a.ttft_ms <= a.total_ms


def test_prefix_cache_hit_on_repeat():
    async def go():
        e = await started()
        first = await e.generate("t", PROMPT, 4)
        second = await e.generate("t", PROMPT, 4)
        await e.stop()
        return first, second

    first, second = run(go())
    assert first.cached_tokens == 0
    assert second.cached_tokens == 96          # 6 full blocks reused


def test_prefix_cache_can_be_disabled_at_runtime():
    async def go():
        e = await started()
        await e.generate("t", PROMPT, 2)
        e.apply_config(e.cfg.merged({"enable_prefix_cache": False}))
        res = await e.generate("t", PROMPT, 2)
        await e.stop()
        return res

    assert run(go()).cached_tokens == 0


def test_batching_beats_serial_execution():
    async def burst(**cfg):
        e = await started(Settings(sim_speed=5.0), **cfg)
        t0 = time.perf_counter()
        await asyncio.gather(*(e.generate("t", PROMPT + f" u{i}", 16) for i in range(16)))
        elapsed = time.perf_counter() - t0
        await e.stop()
        return elapsed

    serial = run(burst(enable_batching=False, enable_prefix_cache=False))
    batched = run(burst(enable_batching=True, enable_prefix_cache=False))
    assert batched < serial / 2, (batched, serial)


def test_queue_full_is_shed():
    async def go():
        e = SimEngine(FAST, RuntimeConfig(max_queue=2))      # engine not started: nothing drains the queue
        e.submit("t", "a", 4)
        e.submit("t", "b", 4)
        try:
            e.submit("t", "c", 4)
        except QueueFull:
            return True
        return False

    assert run(go())


def test_request_larger_than_kv_cache_is_rejected():
    async def go():
        e = SimEngine(Settings(kv_cache_blocks=4), RuntimeConfig())   # 64 tokens of capacity
        try:
            e.submit("t", PROMPT, 4)
        except RequestTooLarge:
            return True
        return False

    assert run(go())


def test_timeout_releases_kv_resources():
    async def go():
        e = await started(Settings(sim_speed=0.01), request_timeout_s=0.5)   # each step takes ~0.6 s
        try:
            await e.generate("t", PROMPT, 10)
        except RequestTimeout:
            pass
        await asyncio.sleep(1.5)
        state = (len(e.running), e.kv._reserved, e.kv._pinned)
        await e.stop()
        return state

    assert run(go()) == (0, 0, 0)


def test_client_cancellation_releases_kv_resources():
    async def go():
        e = await started(Settings(sim_speed=0.05))
        task = asyncio.create_task(e.generate("t", PROMPT, 50))
        await asyncio.sleep(0.4)
        task.cancel()
        await asyncio.sleep(1.0)
        state = (len(e.running), e.kv._reserved, e.kv._pinned)
        await e.stop()
        return state

    assert run(go()) == (0, 0, 0)


def test_streaming_yields_tokens_then_result():
    async def go():
        e = await started()
        pieces, final = [], None
        async for item in e.stream_generate("t", "hello world", 6):
            if isinstance(item, str):
                pieces.append(item)
            else:
                final = item
        await e.stop()
        return pieces, final

    pieces, final = run(go())
    assert len(pieces) == 6 and final.completion_tokens == 6


def test_many_concurrent_requests_leave_no_leaked_blocks():
    async def go():
        e = await started()
        prompts = [" ".join(f"s{i % 3}w{j}" for j in range(80)) + f" u{i}" for i in range(60)]
        for p in prompts[:3]:   # warm-up: requests admitted in the same step cannot share a cache that is not built yet
            await e.generate("t", p, 2)
        results = await asyncio.gather(*(e.generate("t", p, 8 + i % 8) for i, p in enumerate(prompts)))
        state = (len(e.running), len(e.waiting), e.kv._reserved, e.kv._pinned, e.kv.usage())
        hits = sum(r.cached_tokens for r in results)
        await e.stop()
        return state, hits

    (running, waiting, reserved, pinned, usage), hits = run(go())
    assert (running, waiting, reserved, pinned) == (0, 0, 0, 0)
    assert usage <= 1.0 and hits > 0


def test_stats_report_activity():
    async def go():
        e = await started()
        await e.generate("t", PROMPT, 8)
        st = e.stats()
        await e.stop()
        return st

    st = run(go())
    assert st["tokens_generated"] == 8 and st["requests_finished"] == 1 and st["steps"] >= 8
