"""API-key auth with per-tenant rate limits.

``API_KEYS_JSON`` maps each key to a tenant:
    {"sk-team-a": {"tenant": "team-a", "rps": 50, "burst": 100}, "sk-ops": {"tenant": "ops", "admin": true}}
Keys are SHA-256 hashed in memory. Rate limits are token buckets kept per replica; for a global limit across
replicas put a shared limiter (Envoy/Redis) in front.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from .config import Settings


@dataclass(frozen=True)
class Tenant:
    name: str
    rps: float = 0.0     # 0 => unlimited
    burst: float = 0.0
    admin: bool = False


ANONYMOUS = Tenant("anonymous", admin=True)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.auth_enabled
        self._by_hash: dict[str, Tenant] = {}
        self._buckets: dict[str, list[float]] = {}  # tenant -> [tokens, last_refill]
        if self.enabled:
            if not settings.api_keys_json:
                raise RuntimeError("AUTH_ENABLED=true but API_KEYS_JSON is empty")
            for key, spec in json.loads(settings.api_keys_json).items():
                rps = float(spec.get("rps", 0))
                self._by_hash[_hash(key)] = Tenant(
                    name=spec["tenant"], rps=rps, burst=float(spec.get("burst", max(rps, 1))),
                    admin=bool(spec.get("admin", False)),
                )

    def authenticate(self, request: Request) -> Tenant:
        if not self.enabled:
            return ANONYMOUS
        header = request.headers.get("authorization", "")
        key = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        tenant = self._by_hash.get(_hash(key)) if key else None
        if tenant is None:
            raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return tenant

    def check_rate(self, tenant: Tenant) -> None:
        if tenant.rps <= 0:
            return
        now = time.monotonic()
        bucket = self._buckets.setdefault(tenant.name, [tenant.burst, now])
        bucket[0] = min(tenant.burst, bucket[0] + (now - bucket[1]) * tenant.rps)
        bucket[1] = now
        if bucket[0] < 1.0:
            retry = max(1, int((1.0 - bucket[0]) / tenant.rps) + 1)
            raise HTTPException(429, f"rate limit exceeded for tenant {tenant.name}", headers={"Retry-After": str(retry)})
        bucket[0] -= 1.0
