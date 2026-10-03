"""Backends: the built-in simulator, or a proxy to a real OpenAI-compatible server (vLLM, SGLang)."""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

import httpx

from .config import RuntimeConfig, Settings
from .engine import GenResult, QueueFull, RequestTimeout, SimEngine


class UpstreamError(Exception):
    pass


class OpenAIProxyBackend:
    """Forwards to an OpenAI-compatible ``/v1/completions`` endpoint.

    Batching, KV-cache reuse and GPU scheduling happen inside the upstream engine (vLLM/SGLang); this gateway adds
    auth, rate limits, load shedding, metrics and shadow routing in front of it.
    """

    kind = "openai_proxy"

    def __init__(self, settings: Settings, cfg: RuntimeConfig, name: str = "primary", on_step=None) -> None:
        self.settings, self.cfg, self.name = settings, cfg, name
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_url,
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=2000, max_keepalive_connections=200),
        )
        self._model = settings.upstream_model
        self.inflight = 0
        self.requests_finished = 0
        self.tokens_generated = 0

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        await self._client.aclose()

    def apply_config(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self._client.timeout = httpx.Timeout(cfg.request_timeout_s, connect=5.0)

    async def _ensure_model(self) -> str:
        if not self._model:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            self._model = r.json()["data"][0]["id"]
        return self._model

    async def generate(self, tenant: str, prompt: str, max_tokens: int) -> GenResult:
        if self.inflight >= self.cfg.max_queue:
            raise QueueFull(f"too many in-flight requests ({self.cfg.max_queue})")
        self.inflight += 1
        t0 = time.perf_counter()
        try:
            model = await self._ensure_model()
            r = await self._client.post(
                "/v1/completions",
                json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
            )
            r.raise_for_status()
            data = r.json()
        except httpx.TimeoutException:
            raise RequestTimeout("upstream timed out") from None
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise UpstreamError(f"upstream error: {e}") from e
        finally:
            self.inflight -= 1
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        text = data["choices"][0].get("text", "")
        completion = int(usage.get("completion_tokens", 0))
        self.requests_finished += 1
        self.tokens_generated += completion
        return GenResult(
            id=data.get("id") or uuid.uuid4().hex[:16],
            text=text,
            tokens=[],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=completion,
            cached_tokens=int(details.get("cached_tokens") or 0),
            queue_ms=None,
            ttft_ms=None,  # not observable through a non-streaming proxy call
            total_ms=(time.perf_counter() - t0) * 1000,
        )

    async def stream_generate(self, tenant: str, prompt: str, max_tokens: int):
        raise NotImplementedError("streaming is only implemented for the sim backend")
        yield  # pragma: no cover

    def stats(self) -> dict:
        return {
            "waiting": 0,
            "running": self.inflight,
            "kv_usage": 0.0,
            "gpu_util": 0.0,
            "prefix_hit_tokens": 0,
            "prefix_lookup_tokens": 0,
            "tokens_generated": self.tokens_generated,
            "requests_finished": self.requests_finished,
            "steps": 0,
            "batch_sum": 0,
        }


def make_backend(
    settings: Settings,
    cfg: RuntimeConfig,
    name: str = "primary",
    on_step: Optional[Callable[[int, int, float], None]] = None,
):
    if settings.backend == "sim":
        return SimEngine(settings, cfg, name=name, on_step=on_step)
    if settings.backend == "openai_proxy":
        return OpenAIProxyBackend(settings, cfg, name=name)
    raise ValueError(f"unknown BACKEND {settings.backend!r} (expected 'sim' or 'openai_proxy')")
