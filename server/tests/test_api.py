"""API tests. Require fastapi + httpx (`pip install -r requirements-dev.txt`)."""
import json
import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

FAST = dict(sim_speed=50.0)
KEYS = json.dumps({
    "sk-a": {"tenant": "team-a", "rps": 1, "burst": 2},
    "sk-b": {"tenant": "team-b"},
    "sk-admin": {"tenant": "ops", "admin": True},
})


def client(**overrides) -> TestClient:
    return TestClient(create_app(Settings(**{**FAST, **overrides})))


def test_health_and_ready():
    with client() as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/health").status_code == 200
        assert c.get("/readyz").status_code == 200
        info = c.get("/api/info").json()
        assert info["backend"] == "sim" and info["auth_required"] is False


def test_generate_returns_text_usage_and_timings():
    with client() as c:
        r = c.post("/v1/generate", json={"prompt": "hello there general kenobi", "max_tokens": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["usage"]["completion_tokens"] == 8 and len(body["text"].split()) == 8
        assert body["timings"]["total_ms"] > 0
        assert "x-request-id" in r.headers


def test_generate_validation():
    with client() as c:
        assert c.post("/v1/generate", json={"prompt": "", "max_tokens": 8}).status_code == 422
        assert c.post("/v1/generate", json={"prompt": "x", "max_tokens": 0}).status_code == 422


def test_streaming_sse():
    with client() as c:
        with c.stream("POST", "/v1/generate", json={"prompt": "stream me", "max_tokens": 5, "stream": True}) as r:
            assert r.status_code == 200
            lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        events = [json.loads(ln[6:]) for ln in lines[:-1]]
        assert sum(1 for e in events if "text" in e) == 5 and events[-1]["done"] is True


def test_vertex_predict_contract():
    with client() as c:
        r = c.post("/predict", json={"instances": ["first prompt", {"prompt": "second", "max_tokens": 3}],
                                     "parameters": {"max_tokens": 4}})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert [p["usage"]["completion_tokens"] for p in preds] == [4, 3]


def test_oversized_request_is_413():
    with client(kv_cache_blocks=8) as c:    # 128 tokens of KV capacity
        r = c.post("/v1/generate", json={"prompt": " ".join(["w"] * 300), "max_tokens": 8})
        assert r.status_code == 413


def test_metrics_exposed():
    with client() as c:
        c.post("/v1/generate", json={"prompt": "metrics please", "max_tokens": 4})
        text = c.get("/metrics").text
        for name in ("inference_requests_total", "inference_request_duration_seconds_bucket",
                     "inference_queue_depth", "inference_kv_cache_usage_ratio", "gpu_utilization_percent",
                     "inference_batch_size_bucket", "inference_prefix_cache_hit_ratio"):
            assert name in text, name
        assert 'status="ok"' in text


def test_runtime_config_update_and_validation():
    with client() as c:
        assert c.get("/api/config").json()["config"]["max_batch_size"] == 64
        r = c.put("/api/config", json={"max_batch_size": 8, "max_wait_ms": 5})
        assert r.status_code == 200 and r.json()["config"]["max_batch_size"] == 8
        assert c.get("/api/config").json()["config"]["max_wait_ms"] == 5
        assert c.put("/api/config", json={"max_batch_size": 0}).status_code == 422
        assert c.put("/api/config", json={"nonsense": 1}).status_code == 422
        assert c.put("/api/config", json={"shadow_fraction": 0.5}).status_code == 422


def test_stats_and_history_shape():
    with client() as c:
        c.post("/v1/generate", json={"prompt": "warm up", "max_tokens": 4})
        s = c.get("/api/stats").json()
        for key in ("rps", "p50_ms", "p95_ms", "gpu_util", "queue_depth", "kv_usage", "slo"):
            assert key in s
        assert isinstance(c.get("/api/history").json(), list)
        nodes = c.get("/api/cluster").json()
        assert nodes["source"] == "local" and len(nodes["nodes"]) == 1


def test_auth_required_and_admin_only_writes():
    with client(auth_enabled=True, api_keys_json=KEYS) as c:
        assert c.get("/api/info").json()["auth_required"] is True
        assert c.post("/v1/generate", json={"prompt": "x", "max_tokens": 2}).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert c.post("/v1/generate", json={"prompt": "x", "max_tokens": 2}, headers=bad).status_code == 401
        b = {"Authorization": "Bearer sk-b"}
        assert c.post("/v1/generate", json={"prompt": "x", "max_tokens": 2}, headers=b).status_code == 200
        assert c.get("/api/stats", headers=b).status_code == 200
        assert c.put("/api/config", json={"max_batch_size": 4}, headers=b).status_code == 403
        admin = {"X-API-Key": "sk-admin"}
        assert c.put("/api/config", json={"max_batch_size": 4}, headers=admin).status_code == 200


def test_per_tenant_rate_limit():
    with client(auth_enabled=True, api_keys_json=KEYS) as c:
        a = {"Authorization": "Bearer sk-a"}       # 1 rps, burst 2
        codes = [c.post("/v1/generate", json={"prompt": "x", "max_tokens": 1}, headers=a).status_code
                 for _ in range(4)]
        assert codes[:2] == [200, 200] and 429 in codes[2:]
        limited = c.post("/v1/generate", json={"prompt": "x", "max_tokens": 1}, headers=a)
        assert limited.status_code == 429 and "retry-after" in limited.headers
        b = {"Authorization": "Bearer sk-b"}       # other tenants are unaffected
        assert c.post("/v1/generate", json={"prompt": "x", "max_tokens": 1}, headers=b).status_code == 200


def test_auth_enabled_without_keys_fails_fast():
    try:
        create_app(Settings(auth_enabled=True, api_keys_json=""))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_shadow_traffic_compare_and_promote():
    with client() as c:
        r = c.put("/api/shadow", json={"fraction": 1.0, "overrides": {"max_batch_size": 8, "max_wait_ms": 4}})
        assert r.status_code == 200 and r.json()["active"] is True
        for i in range(6):
            assert c.post("/v1/generate", json={"prompt": f"shadow prompt {i}", "max_tokens": 6}).status_code == 200
        deadline, summary = time.time() + 5, {}
        while time.time() < deadline:
            summary = c.get("/api/shadow").json()
            if summary["samples"] >= 6:
                break
            time.sleep(0.1)
        assert summary["samples"] >= 6 and summary["errors"] == 0
        assert summary["output_match_rate"] == 1.0          # same model, same outputs
        assert 'variant="shadow"' in c.get("/metrics").text

        promoted = c.post("/api/shadow/promote")
        assert promoted.status_code == 200 and promoted.json()["config"]["max_batch_size"] == 8
        after = c.get("/api/shadow").json()
        assert after["active"] is False and c.get("/api/config").json()["config"]["shadow_fraction"] == 0.0


def test_shadow_rejects_bad_overrides_and_promote_without_candidate():
    with client() as c:
        assert c.put("/api/shadow", json={"fraction": 0.5, "overrides": {"max_batch_size": -1}}).status_code == 422
        assert c.post("/api/shadow/promote").status_code == 409
        assert c.delete("/api/shadow").json()["active"] is False
