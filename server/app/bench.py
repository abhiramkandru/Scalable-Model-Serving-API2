"""In-process benchmark of the scheduler (no HTTP, no network): `python -m app.bench [closed|sweep|all]`.

Compares three engine configurations on the same workload:
  naive      one sequence at a time, no prefix cache   (a request-at-a-time server)
  batching   continuous batching, no prefix cache
  optimized  continuous batching + prefix/KV-cache reuse

IMPORTANT: this measures the *simulated* GPU cost model in ``Settings`` (decode step = 6 ms + 0.12 ms/seq,
prefill = 0.04 ms/token). The ratios show how the scheduling and caching logic behaves under that model; they are
not measurements of real hardware. Run the same comparison against vLLM on your GPUs before quoting any number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from math import ceil

from .config import RuntimeConfig, Settings
from .engine import QueueFull, RequestTimeout, SimEngine

MODES = {
    "naive": dict(enable_batching=False, enable_prefix_cache=False),
    "batching": dict(enable_batching=True, enable_prefix_cache=False),
    "optimized": dict(enable_batching=True, enable_prefix_cache=True),
}
N_SYSTEM_PROMPTS = 4
SYSTEM_WORDS = 400   # shared prefix length (tokens)
USER_WORDS = 30      # unique suffix length (tokens)


def make_prompts(n: int, rng: random.Random) -> list[tuple[str, int]]:
    systems = [" ".join(f"s{k}w{i}" for i in range(SYSTEM_WORDS)) for k in range(N_SYSTEM_PROMPTS)]
    out = []
    for uid in range(n):
        user = " ".join(f"u{uid}w{i}" for i in range(USER_WORDS))
        out.append((systems[rng.randrange(N_SYSTEM_PROMPTS)] + " " + user, rng.randint(16, 64)))
    return out


def pct(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, ceil(q * len(vals)) - 1))] if vals else 0.0


def summarize(lat_ms: list[float], elapsed: float, engine: SimEngine, errors: dict) -> dict:
    st = engine.stats()
    return {
        "completed": len(lat_ms),
        "throughput_rps": len(lat_ms) / elapsed,
        "tokens_per_s": engine.tokens_generated / elapsed,
        "p50_ms": pct(lat_ms, 0.50),
        "p95_ms": pct(lat_ms, 0.95),
        "p99_ms": pct(lat_ms, 0.99),
        "prefix_hit_rate": st["prefix_hit_tokens"] / st["prefix_lookup_tokens"] if st["prefix_lookup_tokens"] else 0.0,
        "errors": errors,
    }


async def run_closed(mode: str, concurrency: int, duration: float, settings: Settings) -> dict:
    """Closed loop: `concurrency` clients each send a request, wait for the reply, repeat."""
    engine = SimEngine(settings, RuntimeConfig(request_timeout_s=120, **MODES[mode]))
    await engine.start()
    prompts = make_prompts(200_000, random.Random(7))
    counter = iter(range(10**9))
    lat: list[float] = []
    errors = {"shed": 0, "timeout": 0}
    t_end = time.perf_counter() + duration

    async def client():
        while time.perf_counter() < t_end:
            prompt, max_tokens = prompts[next(counter) % len(prompts)]
            try:
                res = await engine.generate("bench", prompt, max_tokens)
                lat.append(res.total_ms)
            except QueueFull:
                errors["shed"] += 1
            except RequestTimeout:
                errors["timeout"] += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(client() for _ in range(concurrency)))
    out = summarize(lat, time.perf_counter() - t0, engine, errors)
    await engine.stop()
    return out


async def run_open(mode: str, rps: float, duration: float, settings: Settings) -> dict:
    """Open loop: Poisson arrivals at `rps` regardless of how fast the server replies (exposes saturation)."""
    engine = SimEngine(settings, RuntimeConfig(request_timeout_s=10, max_queue=2000, **MODES[mode]))
    await engine.start()
    rng = random.Random(11)
    prompts = make_prompts(int(rps * duration * 1.5) + 50, rng)
    lat: list[float] = []
    errors = {"shed": 0, "timeout": 0}

    async def one(prompt, max_tokens):
        try:
            res = await engine.generate("bench", prompt, max_tokens)
            lat.append(res.total_ms)
        except QueueFull:
            errors["shed"] += 1
        except RequestTimeout:
            errors["timeout"] += 1

    tasks = []
    t0 = time.perf_counter()
    t_next = t0
    for prompt, max_tokens in prompts:
        t_next += rng.expovariate(rps)
        if t_next - t0 > duration:
            break
        delay = t_next - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(prompt, max_tokens)))
    await asyncio.gather(*tasks)
    out = summarize(lat, duration, engine, errors)
    out["offered_rps"] = rps
    await engine.stop()
    return out


def fmt_table(rows: list[tuple[str, dict]], first_col: str) -> str:
    head = f"| {first_col} | throughput (req/s) | tokens/s | p50 (ms) | p95 (ms) | p99 (ms) | prefix hit | shed/timeout |\n"
    head += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = ""
    for name, r in rows:
        body += (f"| {name} | {r['throughput_rps']:.1f} | {r['tokens_per_s']:.0f} | {r['p50_ms']:.0f} | "
                 f"{r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {r['prefix_hit_rate']*100:.0f}% | "
                 f"{r['errors']['shed']}/{r['errors']['timeout']} |\n")
    return head + body


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["closed", "sweep", "all"])
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--rates", default="25,50,75,100,125,150")
    ap.add_argument("--json", default="", help="write raw results to this file")
    args = ap.parse_args()
    settings = Settings()
    results: dict = {"cost_model": {
        "decode_base_ms": settings.sim_decode_base_ms, "decode_per_seq_ms": settings.sim_decode_per_seq_ms,
        "prefill_ms_per_token": settings.sim_prefill_ms_per_token,
        "workload": {"shared_prefix_tokens": SYSTEM_WORDS, "unique_suffix_tokens": USER_WORDS,
                     "output_tokens": "16-64 uniform", "distinct_system_prompts": N_SYSTEM_PROMPTS}}}

    if args.what in ("closed", "all"):
        rows = []
        for mode in MODES:
            print(f"closed-loop {mode} ({args.concurrency} clients, {args.duration:.0f}s)...", flush=True)
            rows.append((mode, await run_closed(mode, args.concurrency, args.duration, settings)))
        results["closed_loop"] = {"concurrency": args.concurrency, "duration_s": args.duration, "modes": dict(rows)}
        print(f"\n### Closed loop, {args.concurrency} concurrent clients\n\n" + fmt_table(rows, "mode"))
        base, opt = dict(rows)["naive"], dict(rows)["optimized"]
        print(f"optimized vs naive: {opt['throughput_rps']/base['throughput_rps']:.1f}x throughput, "
              f"p95 latency {100*(opt['p95_ms']/base['p95_ms']-1):+.0f}%\n")

    if args.what in ("sweep", "all"):
        rows = []
        for rate in [float(x) for x in args.rates.split(",")]:
            print(f"open-loop optimized @ {rate:.0f} rps ({args.duration:.0f}s)...", flush=True)
            rows.append((f"{rate:.0f}", await run_open("optimized", rate, args.duration, settings)))
        results["open_loop_optimized"] = {k: v for k, v in rows}
        print("\n### Open loop (Poisson arrivals), optimized mode, single node\n\n" + fmt_table(rows, "offered rps"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"raw results written to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
