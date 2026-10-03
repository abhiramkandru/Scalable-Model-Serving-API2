"""Glue between the API, the backend(s), metrics, shadow traffic and the dashboard snapshots."""
from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from typing import Optional

from .backends import UpstreamError, make_backend
from .config import RuntimeConfig, Settings
from .engine import GenResult, QueueFull, RequestTimeout, RequestTooLarge
from .gpu import GpuSampler
from .metrics import Metrics, RollingStats, _pct, _ratio

MAX_SHADOW_INFLIGHT = 256


class InferenceService:
    def __init__(self, settings: Settings, metrics: Metrics, cfg: Optional[RuntimeConfig] = None) -> None:
        self.settings = settings
        self.metrics = metrics
        self.cfg = cfg or settings.runtime_defaults()
        self.rolling = RollingStats()
        self.history: deque[dict] = deque(maxlen=900)  # 15 min at 1 Hz
        self.latest: Optional[dict] = None
        self.gpu = GpuSampler()
        self.primary = make_backend(settings, self.cfg, "primary", on_step=self._on_step)
        self.shadow = None
        self.shadow_overrides: dict = {}
        self._shadow_pairs: deque[tuple[float, float, bool]] = deque(maxlen=2000)  # primary ms, shadow ms, match
        self._shadow_errors = 0
        self._shadow_inflight = 0
        self._sampler: Optional[asyncio.Task] = None
        self._prev = {"hit": 0, "look": 0, "steps": 0, "batch_sum": 0}
        metrics.bind(self)

    # -- lifecycle -------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self.primary.start()
        self._sampler = asyncio.create_task(self._sample_loop(), name="snapshot-sampler")

    async def stop(self) -> None:
        if self._sampler:
            self._sampler.cancel()
            try:
                await self._sampler
            except asyncio.CancelledError:
                pass
        await self.stop_shadow()
        await self.primary.stop()

    def _on_step(self, n_running: int, prefill_tokens: int, step_s: float) -> None:
        self.metrics.batch_size.observe(n_running)

    # -- request path ----------------------------------------------------------------------------------
    def _record(self, tenant: str, status: str, latency_ms: float, res: Optional[GenResult] = None) -> None:
        m = self.metrics
        m.requests.labels(tenant, status, "primary").inc()
        if status in ("rejected", "rate_limited"):
            return
        self.rolling.record(latency_ms, status, res.completion_tokens if res else 0)
        if res is not None:
            m.duration.labels("primary").observe(res.total_ms / 1000)
            if res.ttft_ms is not None:
                m.ttft.labels("primary").observe(res.ttft_ms / 1000)
            if res.queue_ms is not None:
                m.queue_wait.labels("primary").observe(res.queue_ms / 1000)
            m.prompt_tokens.labels(tenant).inc(res.prompt_tokens)
            m.completion_tokens.labels(tenant).inc(res.completion_tokens)
            m.cached_tokens.labels(tenant).inc(res.cached_tokens)

    async def generate(self, tenant: str, prompt: str, max_tokens: int) -> GenResult:
        t0 = time.perf_counter()
        try:
            res = await self.primary.generate(tenant, prompt, max_tokens)
        except QueueFull:
            self._record(tenant, "shed", (time.perf_counter() - t0) * 1000)
            raise
        except RequestTimeout:
            self._record(tenant, "timeout", (time.perf_counter() - t0) * 1000)
            raise
        except RequestTooLarge:
            self._record(tenant, "rejected", 0)
            raise
        except asyncio.CancelledError:
            raise  # client went away; not a server error
        except Exception:
            self._record(tenant, "error", (time.perf_counter() - t0) * 1000)
            raise
        self._record(tenant, "ok", res.total_ms, res)
        self._mirror(tenant, prompt, max_tokens, res)
        return res

    async def stream(self, tenant: str, prompt: str, max_tokens: int):
        """Yields text pieces then the final GenResult; records metrics like ``generate``."""
        t0 = time.perf_counter()
        try:
            async for item in self.primary.stream_generate(tenant, prompt, max_tokens):
                if isinstance(item, GenResult):
                    self._record(tenant, "ok", item.total_ms, item)
                    self._mirror(tenant, prompt, max_tokens, item)
                yield item
        except QueueFull:
            self._record(tenant, "shed", (time.perf_counter() - t0) * 1000)
            raise
        except RequestTimeout:
            self._record(tenant, "timeout", (time.perf_counter() - t0) * 1000)
            raise
        except RequestTooLarge:
            self._record(tenant, "rejected", 0)
            raise

    # -- shadow traffic --------------------------------------------------------------------------------
    def _shadow_cfg(self) -> RuntimeConfig:
        return self.cfg.merged({**self.shadow_overrides, "shadow_fraction": 0.0})

    def _mirror(self, tenant: str, prompt: str, max_tokens: int, primary_res: GenResult) -> None:
        if self.shadow is None or self.cfg.shadow_fraction <= 0 or random.random() >= self.cfg.shadow_fraction:
            return
        if self._shadow_inflight >= MAX_SHADOW_INFLIGHT:
            self.metrics.shadow_dropped.inc()
            return
        self._shadow_inflight += 1
        asyncio.create_task(self._run_shadow(self.shadow, tenant, prompt, max_tokens, primary_res))

    async def _run_shadow(self, backend, tenant: str, prompt: str, max_tokens: int, primary_res: GenResult) -> None:
        m = self.metrics
        try:
            res = await backend.generate(f"shadow:{tenant}", prompt, max_tokens)
            m.requests.labels(tenant, "ok", "shadow").inc()
            m.duration.labels("shadow").observe(res.total_ms / 1000)
            if res.ttft_ms is not None:
                m.ttft.labels("shadow").observe(res.ttft_ms / 1000)
            same = (res.tokens == primary_res.tokens) if res.tokens and primary_res.tokens else res.text == primary_res.text
            self._shadow_pairs.append((primary_res.total_ms, res.total_ms, same))
        except asyncio.CancelledError:
            raise
        except Exception:
            self._shadow_errors += 1
            m.requests.labels(tenant, "error", "shadow").inc()
        finally:
            self._shadow_inflight -= 1

    async def set_shadow(self, fraction: float, overrides: dict) -> None:
        overrides = {k: v for k, v in overrides.items() if k != "shadow_fraction"}
        self.cfg.merged({**overrides, "shadow_fraction": fraction})  # validate before touching state
        self.shadow_overrides = overrides
        if self.shadow is None:
            self.shadow = make_backend(self.settings, self._shadow_cfg(), "shadow")
            await self.shadow.start()
            self._shadow_pairs.clear()
            self._shadow_errors = 0
        self.apply_config(self.cfg.merged({"shadow_fraction": fraction}))

    async def stop_shadow(self) -> None:
        shadow, self.shadow = self.shadow, None
        self.shadow_overrides = {}
        if shadow is not None:
            await shadow.stop()
        if self.cfg.shadow_fraction:
            self.apply_config(self.cfg.merged({"shadow_fraction": 0.0}))

    async def promote_shadow(self) -> RuntimeConfig:
        if self.shadow is None:
            raise ValueError("no shadow candidate is running")
        new_cfg = self.cfg.merged({**self.shadow_overrides, "shadow_fraction": 0.0})
        await self.stop_shadow()
        self.apply_config(new_cfg)
        return new_cfg

    def shadow_summary(self) -> dict:
        pairs = list(self._shadow_pairs)
        prim = sorted(p for p, _, _ in pairs)
        shad = sorted(s for _, s, _ in pairs)
        n = len(pairs)

        def delta(a: float, b: float) -> Optional[float]:
            return (b - a) / a * 100 if a else None

        p95p, p95s = _pct(prim, 0.95), _pct(shad, 0.95)
        return {
            "active": self.shadow is not None,
            "fraction": self.cfg.shadow_fraction,
            "overrides": self.shadow_overrides,
            "samples": n,
            "errors": self._shadow_errors,
            "dropped_inflight_cap": MAX_SHADOW_INFLIGHT,
            "primary": {"p50_ms": _pct(prim, 0.5), "p95_ms": p95p},
            "shadow": {"p50_ms": _pct(shad, 0.5), "p95_ms": p95s},
            "p95_delta_pct": delta(p95p, p95s),
            "output_match_rate": (sum(1 for *_, same in pairs if same) / n) if n else None,
        }

    # -- config ----------------------------------------------------------------------------------------
    def apply_config(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.primary.apply_config(cfg)
        if self.shadow is not None:
            self.shadow.apply_config(self._shadow_cfg())

    # -- telemetry -------------------------------------------------------------------------------------
    def gpu_util(self) -> float:
        real = self.gpu.read()
        return real["util"] if real else self.primary.stats()["gpu_util"]

    def _snapshot(self, update_prev: bool) -> dict:
        st = self.primary.stats()
        roll = self.rolling.window(10.0)
        d_hit = st["prefix_hit_tokens"] - self._prev["hit"]
        d_look = st["prefix_lookup_tokens"] - self._prev["look"]
        d_steps = st["steps"] - self._prev["steps"]
        d_batch = st["batch_sum"] - self._prev["batch_sum"]
        if update_prev:
            self._prev = {"hit": st["prefix_hit_tokens"], "look": st["prefix_lookup_tokens"],
                          "steps": st["steps"], "batch_sum": st["batch_sum"]}
        real = self.gpu.read()
        if real:
            util, source = real["util"], "nvml"
        elif self.primary.kind == "sim":
            util, source = st["gpu_util"], "simulated"
        else:
            util, source = 0.0, "n/a"
        return {
            "ts": time.time(),
            **roll,
            "gpu_util": util,
            "gpu_source": source,
            "queue_depth": st["waiting"],
            "running": st["running"],
            "batch_size": (d_batch / d_steps) if d_steps else float(st["running"]),
            "kv_usage": st["kv_usage"],
            "prefix_hit_rate": (d_hit / d_look) if d_look else (_ratio(st) if not update_prev else 0.0),
            "shadow_fraction": self.cfg.shadow_fraction,
            "slo": self.rolling.slo(120.0, self.settings.slo_availability, self.settings.slo_latency_ms),
        }

    def snapshot(self) -> dict:
        return self.latest or self._snapshot(update_prev=False)

    async def _sample_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.latest = self._snapshot(update_prev=True)
            self.history.append(self.latest)
