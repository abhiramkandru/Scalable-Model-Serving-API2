"""Prometheus metrics plus a rolling window used for the dashboard and the SLO panel."""
from __future__ import annotations

import time
from collections import deque
from math import ceil
from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30)
BAD_STATUSES = {"error", "timeout", "shed"}  # count against the availability SLO (4xx are the caller's fault)


class Metrics:
    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "inference_requests_total", "Inference requests by outcome", ["tenant", "status", "variant"], registry=r
        )
        self.duration = Histogram(
            "inference_request_duration_seconds", "End-to-end request latency", ["variant"],
            buckets=LATENCY_BUCKETS, registry=r,
        )
        self.ttft = Histogram(
            "inference_time_to_first_token_seconds", "Time to first token", ["variant"],
            buckets=LATENCY_BUCKETS, registry=r,
        )
        self.queue_wait = Histogram(
            "inference_queue_wait_seconds", "Time spent waiting for admission", ["variant"],
            buckets=LATENCY_BUCKETS, registry=r,
        )
        self.prompt_tokens = Counter("inference_prompt_tokens_total", "Prompt tokens", ["tenant"], registry=r)
        self.completion_tokens = Counter(
            "inference_completion_tokens_total", "Generated tokens", ["tenant"], registry=r
        )
        self.cached_tokens = Counter(
            "inference_cached_prompt_tokens_total", "Prompt tokens served from the prefix cache", ["tenant"],
            registry=r,
        )
        self.batch_size = Histogram(
            "inference_batch_size", "Sequences per decode step",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), registry=r,
        )
        self.shadow_dropped = Counter(
            "inference_shadow_dropped_total", "Mirrored requests dropped (shadow backend saturated)", registry=r
        )

    def bind(self, service) -> None:
        """Gauges that are read from live state at scrape time."""
        r = self.registry
        Gauge("inference_queue_depth", "Requests waiting for admission", registry=r).set_function(
            lambda: service.primary.stats()["waiting"]
        )
        Gauge("inference_running_sequences", "Sequences in the running batch", registry=r).set_function(
            lambda: service.primary.stats()["running"]
        )
        Gauge("inference_kv_cache_usage_ratio", "Fraction of KV-cache blocks in use", registry=r).set_function(
            lambda: service.primary.stats()["kv_usage"]
        )
        Gauge("inference_prefix_cache_hit_ratio", "Cumulative prefix-cache token hit ratio", registry=r).set_function(
            lambda: _ratio(service.primary.stats())
        )
        Gauge("gpu_utilization_percent", "GPU utilisation (NVML, or simulated)", registry=r).set_function(
            service.gpu_util
        )


def _ratio(st: dict) -> float:
    return st["prefix_hit_tokens"] / st["prefix_lookup_tokens"] if st["prefix_lookup_tokens"] else 0.0


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, max(0, ceil(q * len(sorted_vals)) - 1))]


class RollingStats:
    """Recent request outcomes kept in memory (not a metrics store; Prometheus remains the source of truth)."""

    def __init__(self, retention_s: float = 120.0) -> None:
        self.retention_s = retention_s
        self.events: deque[tuple[float, float, str, int]] = deque()  # (ts, latency_ms, status, completion_tokens)

    def record(self, latency_ms: float, status: str, completion_tokens: int = 0) -> None:
        now = time.time()
        self.events.append((now, latency_ms, status, completion_tokens))
        cutoff = now - self.retention_s
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def window(self, seconds: float) -> dict:
        cutoff = time.time() - seconds
        lat: list[float] = []
        total = bad = tokens = 0
        for ts, ms, status, toks in reversed(self.events):
            if ts < cutoff:
                break
            total += 1
            if status in BAD_STATUSES:
                bad += 1
            else:
                lat.append(ms)
                tokens += toks
        lat.sort()
        return {
            "rps": total / seconds,
            "p50_ms": _pct(lat, 0.50),
            "p95_ms": _pct(lat, 0.95),
            "p99_ms": _pct(lat, 0.99),
            "error_rate": bad / total if total else 0.0,
            "tokens_per_s": tokens / seconds,
        }

    def slo(self, seconds: float, target: float, latency_ms: float) -> dict:
        cutoff = time.time() - seconds
        total = good = 0
        for ts, ms, status, _ in reversed(self.events):
            if ts < cutoff:
                break
            total += 1
            if status not in BAD_STATUSES and ms <= latency_ms:
                good += 1
        ratio = good / total if total else 1.0
        burn = (1.0 - ratio) / (1.0 - target) if target < 1.0 else 0.0
        return {
            "target": target,
            "latency_ms": latency_ms,
            "window_s": seconds,
            "requests": total,
            "good_ratio": ratio,
            "burn_rate": burn,  # 1.0 == consuming budget exactly as fast as the SLO allows
        }
