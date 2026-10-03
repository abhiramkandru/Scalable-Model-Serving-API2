#!/usr/bin/env python3
"""Generate deploy/grafana/dashboards/inference.json.   python deploy/grafana/gen_dashboard.py"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
P = 'variant="primary"'


def hq(q: str, variant: str = "primary", window: str = "1m", metric: str = "inference_request_duration_seconds") -> str:
    return (f'histogram_quantile({q}, sum by (le) (rate({metric}_bucket{{variant="{variant}"}}[{window}])))')


def targets(items):
    return [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": DS} for i, (e, l) in enumerate(items)]


def ts(title, items, x, y, w=8, h=8, unit="short", **extra):
    defaults = {"unit": unit, **extra}
    return {
        "type": "timeseries", "title": title, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
        "targets": targets(items),
    }


def stat(title, expr, x, unit="short", steps=None, decimals=None):
    steps = steps or [{"color": "green", "value": None}]
    d = {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps}}
    if decimals is not None:
        d["decimals"] = decimals
    return {
        "type": "stat", "title": title, "datasource": DS, "gridPos": {"x": x, "y": 0, "w": 4, "h": 4},
        "fieldConfig": {"defaults": d, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"},
        "targets": targets([(expr, "")]),
    }


ERR = f'sum(rate(inference_requests_total{{{P},status=~"error|timeout|shed"}}[5m])) / sum(rate(inference_requests_total{{{P},status=~"ok|error|timeout|shed"}}[5m]))'

panels = [
    stat("Requests / s", f"sum(rate(inference_requests_total{{{P}}}[1m]))", 0, decimals=1),
    stat("p95 latency", hq(0.95), 4, "s", [{"color": "green", "value": None}, {"color": "orange", "value": 0.5}, {"color": "red", "value": 1}]),
    stat("Error ratio (5m)", ERR, 8, "percentunit", [{"color": "green", "value": None}, {"color": "orange", "value": 0.005}, {"color": "red", "value": 0.02}], decimals=2),
    stat("GPU util (avg)", "avg(gpu_utilization_percent)", 12, "percent", decimals=0),
    stat("Queue depth", "sum(inference_queue_depth)", 16, steps=[{"color": "green", "value": None}, {"color": "orange", "value": 50}, {"color": "red", "value": 100}]),
    stat("Tokens / s", "sum(rate(inference_completion_tokens_total[1m]))", 20, decimals=0),

    ts("Latency", [(hq(0.5), "p50"), (hq(0.95), "p95"), (hq(0.99), "p99")], 0, 4, 12, unit="s"),
    ts("Time to first token", [(hq(0.5, metric="inference_time_to_first_token_seconds"), "p50"),
                               (hq(0.95, metric="inference_time_to_first_token_seconds"), "p95")], 12, 4, 12, unit="s"),

    ts("Throughput", [(f"sum(rate(inference_requests_total{{{P}}}[1m]))", "req/s"),
                      ("sum(rate(inference_completion_tokens_total[1m]))", "tokens/s")], 0, 12),
    ts("Requests by outcome", [(f"sum by (status) (rate(inference_requests_total{{{P}}}[1m]))", "{{status}}")], 8, 12),
    ts("Requests by tenant", [(f"sum by (tenant) (rate(inference_requests_total{{{P}}}[1m]))", "{{tenant}}")], 16, 12),

    ts("GPU utilisation", [("avg(gpu_utilization_percent)", "app / NVML (avg)"),
                           ("avg(DCGM_FI_DEV_GPU_UTIL)", "DCGM (GKE managed)")], 0, 20, unit="percent", min=0, max=100),
    ts("Queue & running batch", [("sum(inference_queue_depth)", "queued"), ("sum(inference_running_sequences)", "running"),
                                 ("sum(rate(inference_batch_size_sum[1m])) / sum(rate(inference_batch_size_count[1m]))", "avg batch")], 8, 20),
    ts("KV cache & prefix reuse", [("avg(inference_kv_cache_usage_ratio)", "KV cache usage"),
                                   ("avg(inference_prefix_cache_hit_ratio)", "prefix hit ratio")], 16, 20, unit="percentunit", min=0, max=1),

    ts("Error-budget burn rate (1.0 = on budget)", [
        ("job:inference_error_ratio:rate5m / 0.005", "5m"), ("job:inference_error_ratio:rate1h / 0.005", "1h"),
        ("job:inference_error_ratio:rate6h / 0.005", "6h")], 0, 28, 12),
    ts("Shadow vs primary p95", [(hq(0.95, "primary", "5m"), "primary"), (hq(0.95, "shadow", "5m"), "shadow candidate")], 12, 28, 12, unit="s"),
]

dashboard = {
    "uid": "model-serving", "title": "Model serving", "tags": ["inference", "slo"], "schemaVersion": 39, "version": 1,
    "editable": True, "refresh": "10s", "time": {"from": "now-30m", "to": "now"}, "templating": {"list": []}, "panels": panels,
}
for i, p in enumerate(panels, 1):
    p["id"] = i

out = Path(__file__).resolve().parent / "dashboards" / "inference.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print("wrote", out, f"({len(panels)} panels)")
