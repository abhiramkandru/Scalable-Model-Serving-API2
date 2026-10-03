#!/usr/bin/env python3
"""Open-loop HTTP load generator for the serving API.

    python loadtest/loadtest.py --url http://localhost:8000 --rps 80 --duration 30
    python loadtest/loadtest.py --url https://gateway.example.com --api-key sk-... --rps 500 --duration 60

Requests share a long system prompt (so prefix caching has something to reuse) plus a unique suffix.
Arrivals are Poisson at --rps regardless of response times, so overload shows up as latency/errors.
Reports achieved RPS, latency percentiles and a status-code breakdown. Needs `httpx`.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from collections import Counter
from math import ceil

import httpx

SYSTEM_PROMPTS = [" ".join(f"s{k}w{i}" for i in range(400)) for k in range(4)]


def pct(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, ceil(q * len(vals)) - 1))] if vals else 0.0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=float, default=50)
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-tokens-min", type=int, default=16)
    ap.add_argument("--max-tokens-max", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    lat: list[float] = []
    codes: Counter = Counter()

    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=args.timeout, limits=limits) as client:

        async def one(i: int) -> None:
            prompt = f"{rng.choice(SYSTEM_PROMPTS)} " + " ".join(f"u{i}w{j}" for j in range(30))
            body = {"prompt": prompt, "max_tokens": rng.randint(args.max_tokens_min, args.max_tokens_max)}
            t0 = time.perf_counter()
            try:
                r = await client.post("/v1/generate", json=body)
                codes[r.status_code] += 1
                if r.status_code == 200:
                    lat.append((time.perf_counter() - t0) * 1000)
            except httpx.HTTPError as e:
                codes[type(e).__name__] += 1

        tasks = []
        t0 = time.perf_counter()
        t_next, i = t0, 0
        while True:
            t_next += rng.expovariate(args.rps)
            if t_next - t0 > args.duration:
                break
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i)))
            i += 1
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    total = sum(codes.values())
    print(f"sent {total} requests in {elapsed:.1f}s (offered {args.rps:.0f} rps, achieved {len(lat)/args.duration:.1f} ok/s)")
    print(f"latency ms: p50={pct(lat,.5):.0f} p95={pct(lat,.95):.0f} p99={pct(lat,.99):.0f} max={max(lat, default=0):.0f}")
    print("status codes:", dict(sorted(codes.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    asyncio.run(main())
