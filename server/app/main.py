"""FastAPI application: inference endpoints, Vertex AI contract, runtime control plane, metrics."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .auth import Authenticator, Tenant
from .backends import UpstreamError
from .config import RuntimeConfig, RuntimeConfigPatch, Settings
from .engine import GenResult, QueueFull, RequestTimeout, RequestTooLarge
from .metrics import Metrics
from .service import InferenceService

log = logging.getLogger("serving")
router = APIRouter()


# -- request models ------------------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=200_000)
    max_tokens: int = Field(64, ge=1, le=4096)
    stream: bool = False


class PredictRequest(BaseModel):
    """Vertex AI custom-container request: {"instances": [...], "parameters": {...}}."""
    instances: list = Field(..., min_length=1, max_length=64)
    parameters: dict = Field(default_factory=dict)


class ShadowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fraction: float = Field(0.1, ge=0.0, le=1.0)
    overrides: dict = Field(default_factory=dict)


# -- dependencies --------------------------------------------------------------------------------------
def current_tenant(request: Request) -> Tenant:
    return request.app.state.auth.authenticate(request)


def admin_tenant(tenant: Tenant = Depends(current_tenant)) -> Tenant:
    if not tenant.admin:
        raise HTTPException(403, "admin API key required")
    return tenant


def limited_tenant(request: Request, tenant: Tenant = Depends(current_tenant)) -> Tenant:
    try:
        request.app.state.auth.check_rate(tenant)
    except HTTPException:
        request.app.state.metrics.requests.labels(tenant.name, "rate_limited", "primary").inc()
        raise
    return tenant


def _result_json(res: GenResult) -> dict:
    return {
        "id": res.id,
        "text": res.text,
        "usage": {
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "cached_tokens": res.cached_tokens,
        },
        "timings": {
            "queue_ms": res.queue_ms,
            "ttft_ms": res.ttft_ms,
            "total_ms": round(res.total_ms, 3),
        },
    }


def _http_error(e: Exception) -> HTTPException:
    if isinstance(e, QueueFull):
        return HTTPException(503, str(e), headers={"Retry-After": "1"})
    if isinstance(e, RequestTimeout):
        return HTTPException(504, str(e))
    if isinstance(e, RequestTooLarge):
        return HTTPException(413, str(e))
    if isinstance(e, UpstreamError):
        return HTTPException(502, str(e))
    if isinstance(e, NotImplementedError):
        return HTTPException(501, str(e))
    raise e


# -- inference endpoints -------------------------------------------------------------------------------
@router.post("/v1/generate")
async def generate(body: GenerateRequest, request: Request, tenant: Tenant = Depends(limited_tenant)):
    svc: InferenceService = request.app.state.service
    if body.stream:
        gen = svc.stream(tenant.name, body.prompt, body.max_tokens)
        try:  # pull the first item so admission errors become proper HTTP statuses, not a broken stream
            first = await gen.__anext__()
        except (QueueFull, RequestTimeout, RequestTooLarge, UpstreamError, NotImplementedError) as e:
            raise _http_error(e)

        async def events():
            item = first
            try:
                while True:
                    if isinstance(item, GenResult):
                        yield f"data: {json.dumps({'done': True, **_result_json(item)})}\n\n"
                    else:
                        yield f"data: {json.dumps({'text': item})}\n\n"
                    item = await gen.__anext__()
            except StopAsyncIteration:
                yield "data: [DONE]\n\n"
            except (QueueFull, RequestTimeout, RequestTooLarge, UpstreamError) as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                await gen.aclose()

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    try:
        res = await svc.generate(tenant.name, body.prompt, body.max_tokens)
    except (QueueFull, RequestTimeout, RequestTooLarge, UpstreamError) as e:
        raise _http_error(e)
    return _result_json(res)


async def predict(body: PredictRequest, request: Request, tenant: Tenant = Depends(limited_tenant)):
    """Vertex AI prediction route: each instance is a prompt string or {"prompt": ..., "max_tokens": ...}."""
    svc: InferenceService = request.app.state.service
    default_max = int(body.parameters.get("max_tokens", 64))

    async def one(inst):
        prompt, max_tokens = (inst, default_max) if isinstance(inst, str) else (
            str(inst.get("prompt", "")), int(inst.get("max_tokens", default_max)))
        if not prompt or not 1 <= max_tokens <= 4096:
            raise HTTPException(422, "each instance needs a non-empty prompt and 1 <= max_tokens <= 4096")
        return _result_json(await svc.generate(tenant.name, prompt, max_tokens))

    try:
        return {"predictions": list(await asyncio.gather(*(one(i) for i in body.instances)))}
    except (QueueFull, RequestTimeout, RequestTooLarge, UpstreamError) as e:
        raise _http_error(e)


# -- health / metrics / info ---------------------------------------------------------------------------
async def healthz():
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    svc: InferenceService = request.app.state.service
    ok = getattr(svc.primary, "_task", True) is not None
    return JSONResponse({"status": "ready" if ok else "starting"}, status_code=200 if ok else 503)


@router.get("/metrics")
async def metrics_endpoint(request: Request):
    return Response(generate_latest(request.app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/info")
async def info(request: Request):
    s: Settings = request.app.state.settings
    return {
        "version": __version__,
        "node": s.node_name,
        "backend": request.app.state.service.primary.kind,
        "auth_required": s.auth_enabled,
        "gpu_source": "nvml" if request.app.state.service.gpu.available else (
            "simulated" if s.backend == "sim" else "n/a"),
    }


# -- dashboard / control plane -------------------------------------------------------------------------
@router.get("/api/me")
async def me(tenant: Tenant = Depends(current_tenant)):
    return {"tenant": tenant.name, "admin": tenant.admin}


@router.get("/api/stats")
async def stats(request: Request, tenant: Tenant = Depends(current_tenant)):
    return request.app.state.service.snapshot()


@router.get("/api/history")
async def history(request: Request, tenant: Tenant = Depends(current_tenant)):
    return list(request.app.state.service.history)


@router.get("/api/config")
async def get_config(request: Request, tenant: Tenant = Depends(current_tenant)):
    s: Settings = request.app.state.settings
    return {
        "config": request.app.state.service.cfg.model_dump(),
        "startup": {"backend": s.backend, "kv_cache_blocks": s.kv_cache_blocks, "kv_block_size": s.kv_block_size},
    }


@router.put("/api/config")
async def put_config(patch: RuntimeConfigPatch, request: Request, tenant: Tenant = Depends(admin_tenant)):
    svc: InferenceService = request.app.state.service
    changes = patch.changes()
    if "shadow_fraction" in changes:
        raise HTTPException(422, "use PUT /api/shadow to control shadow traffic")
    try:
        new_cfg = svc.cfg.merged(changes)
    except ValueError as e:
        raise HTTPException(422, str(e))
    svc.apply_config(new_cfg)
    log.info(json.dumps({"event": "config_update", "tenant": tenant.name, "changes": changes}))
    return {"config": new_cfg.model_dump()}


@router.get("/api/shadow")
async def get_shadow(request: Request, tenant: Tenant = Depends(current_tenant)):
    return request.app.state.service.shadow_summary()


@router.put("/api/shadow")
async def put_shadow(body: ShadowRequest, request: Request, tenant: Tenant = Depends(admin_tenant)):
    svc: InferenceService = request.app.state.service
    try:
        await svc.set_shadow(body.fraction, body.overrides)
    except ValueError as e:
        raise HTTPException(422, str(e))
    log.info(json.dumps({"event": "shadow_update", "tenant": tenant.name, "fraction": body.fraction,
                         "overrides": body.overrides}))
    return svc.shadow_summary()


@router.delete("/api/shadow")
async def delete_shadow(request: Request, tenant: Tenant = Depends(admin_tenant)):
    await request.app.state.service.stop_shadow()
    return request.app.state.service.shadow_summary()


@router.post("/api/shadow/promote")
async def promote_shadow(request: Request, tenant: Tenant = Depends(admin_tenant)):
    svc: InferenceService = request.app.state.service
    try:
        cfg = await svc.promote_shadow()
    except ValueError as e:
        raise HTTPException(409, str(e))
    log.info(json.dumps({"event": "shadow_promoted", "tenant": tenant.name}))
    return {"config": cfg.model_dump()}


_CLUSTER_QUERIES = {
    "gpu_util": "gpu_utilization_percent",
    "queue": "inference_queue_depth",
    "running": "inference_running_sequences",
    "kv": "inference_kv_cache_usage_ratio",
    "rps": 'sum by (instance)(rate(inference_requests_total{variant="primary"}[1m]))',
    "p95_ms": 'histogram_quantile(0.95, sum by (instance, le)'
              '(rate(inference_request_duration_seconds_bucket{variant="primary"}[1m]))) * 1000',
}


@router.get("/api/cluster")
async def cluster(request: Request, tenant: Tenant = Depends(current_tenant)):
    """Per-node view from Prometheus (all replicas); falls back to this node when PROMETHEUS_URL is unset."""
    s: Settings = request.app.state.settings
    svc: InferenceService = request.app.state.service
    if not s.prometheus_url:
        snap = svc.snapshot()
        return {"source": "local", "nodes": [{
            "node": s.node_name, "gpu_util": snap["gpu_util"], "queue": snap["queue_depth"],
            "running": snap["running"], "kv": snap["kv_usage"], "rps": snap["rps"], "p95_ms": snap["p95_ms"]}]}
    nodes: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            results = await asyncio.gather(*(
                client.get(f"{s.prometheus_url}/api/v1/query", params={"query": q}) for q in _CLUSTER_QUERIES.values()))
        for key, resp in zip(_CLUSTER_QUERIES, results):
            resp.raise_for_status()
            for row in resp.json()["data"]["result"]:
                inst = row["metric"].get("instance", "?")
                val = float(row["value"][1])
                nodes.setdefault(inst, {"node": inst})[key] = None if val != val else val  # NaN -> null
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise HTTPException(502, f"prometheus query failed: {e}")
    return {"source": "prometheus", "nodes": sorted(nodes.values(), key=lambda n: n["node"])}


# -- app factory ---------------------------------------------------------------------------------------
def _maybe_instrument_otel(app: FastAPI) -> None:
    """Distributed tracing: enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set and the OTel packages exist."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "model-gateway")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz,readyz")
    except ImportError:
        log.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but OpenTelemetry packages are not installed")


def create_app(settings: Optional[Settings] = None, cfg: Optional[RuntimeConfig] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()
    service = InferenceService(settings, metrics, cfg)
    auth = Authenticator(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="Scalable Model Serving API", version=__version__, lifespan=lifespan)
    app.state.settings, app.state.metrics, app.state.service, app.state.auth = settings, metrics, service, auth

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
                           allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        if request.url.path not in ("/metrics", "/healthz", "/readyz", "/health", "/api/stats", "/api/history"):
            log.info(json.dumps({"event": "request", "id": rid, "method": request.method, "path": request.url.path,
                                 "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        return response

    app.include_router(router)
    for path in {"/predict", os.getenv("AIP_PREDICT_ROUTE") or "/predict"}:  # Vertex AI custom-container contract
        app.add_api_route(path, predict, methods=["POST"])
    for path in {"/healthz", "/health", os.getenv("AIP_HEALTH_ROUTE") or "/health"}:
        app.add_api_route(path, healthz, methods=["GET"])
    _maybe_instrument_otel(app)
    return app


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
app = create_app()
