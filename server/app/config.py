"""Hot-reloadable runtime config and startup-only settings."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    """Settings that can be changed while the server is running (PUT /api/config)."""

    model_config = ConfigDict(extra="forbid")

    enable_batching: bool = True          # False => one sequence at a time (naive baseline)
    enable_prefix_cache: bool = True      # KV-cache reuse across requests sharing a prompt prefix
    max_batch_size: int = Field(64, ge=1, le=1024)
    max_batch_tokens: int = Field(16384, ge=16, le=262144)   # prefill-token budget per scheduler step
    max_wait_ms: float = Field(2.0, ge=0, le=500)            # batching window when the engine is idle
    max_queue: int = Field(1000, ge=1, le=100_000)           # beyond this, requests are shed (503)
    request_timeout_s: float = Field(30.0, ge=0.5, le=600)
    shadow_fraction: float = Field(0.0, ge=0.0, le=1.0)      # share of traffic mirrored to the shadow backend

    def merged(self, patch: dict) -> "RuntimeConfig":
        return RuntimeConfig.model_validate({**self.model_dump(), **patch})


class RuntimeConfigPatch(BaseModel):
    """Partial update; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    enable_batching: Optional[bool] = None
    enable_prefix_cache: Optional[bool] = None
    max_batch_size: Optional[int] = None
    max_batch_tokens: Optional[int] = None
    max_wait_ms: Optional[float] = None
    max_queue: Optional[int] = None
    request_timeout_s: Optional[float] = None
    shadow_fraction: Optional[float] = None

    def changes(self) -> dict:
        return self.model_dump(exclude_none=True)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Startup-only settings, read from the environment."""

    backend: str = "sim"                       # "sim" | "openai_proxy"
    upstream_url: str = "http://vllm:8000"     # OpenAI-compatible server (vLLM / SGLang) for openai_proxy
    upstream_model: str = ""                   # empty => discover via /v1/models
    kv_cache_blocks: int = 4096                # simulated KV capacity (blocks)
    kv_block_size: int = 16                    # tokens per KV block
    sim_prefill_ms_per_token: float = 0.04
    sim_decode_base_ms: float = 6.0            # per decode step, independent of batch size (weight loading)
    sim_decode_per_seq_ms: float = 0.12        # marginal cost of each extra sequence in the batch
    sim_speed: float = 1.0                     # >1 runs the simulation faster than real time (tests only)
    auth_enabled: bool = False
    api_keys_json: str = ""
    slo_availability: float = 0.995
    slo_latency_ms: float = 500.0
    prometheus_url: str = ""
    node_name: str = ""
    cors_origins: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.getenv
        return cls(
            backend=e("BACKEND", "sim"),
            upstream_url=e("UPSTREAM_URL", "http://vllm:8000"),
            upstream_model=e("UPSTREAM_MODEL", ""),
            kv_cache_blocks=int(e("KV_CACHE_BLOCKS", "4096")),
            kv_block_size=int(e("KV_BLOCK_SIZE", "16")),
            sim_prefill_ms_per_token=float(e("SIM_PREFILL_MS_PER_TOKEN", "0.04")),
            sim_decode_base_ms=float(e("SIM_DECODE_BASE_MS", "6.0")),
            sim_decode_per_seq_ms=float(e("SIM_DECODE_PER_SEQ_MS", "0.12")),
            sim_speed=float(e("SIM_SPEED", "1.0")),
            auth_enabled=_env_bool("AUTH_ENABLED", False),
            api_keys_json=e("API_KEYS_JSON", ""),
            slo_availability=float(e("SLO_AVAILABILITY", "0.995")),
            slo_latency_ms=float(e("SLO_LATENCY_MS", "500")),
            prometheus_url=e("PROMETHEUS_URL", "").rstrip("/"),
            node_name=e("NODE_NAME") or e("HOSTNAME") or socket.gethostname(),
            cors_origins=e("CORS_ORIGINS", ""),
        )

    def runtime_defaults(self) -> RuntimeConfig:
        return RuntimeConfig()
