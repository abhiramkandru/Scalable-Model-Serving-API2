#!/usr/bin/env python3
"""Upload a serving container to Vertex AI and deploy it to an endpoint with GPU autoscaling.

NOT TESTED against a live project. Requires `pip install google-cloud-aiplatform` and application-default credentials.

Gateway image (this repo's server/, speaks the Vertex custom-container contract on /predict and /health):
    python deploy/vertex/deploy.py --project P --region us-central1 \
        --image us-central1-docker.pkg.dev/P/serving/gateway:1.0.0 --env BACKEND=sim

A model server image that already speaks its own routes (e.g. vLLM's OpenAI-compatible server):
    python deploy/vertex/deploy.py --project P --image <vllm-image> \
        --predict-route /v1/completions --health-route /health --port 8000 \
        --env HUGGING_FACE_HUB_TOKEN=... --args --model=Qwen/Qwen2.5-7B-Instruct --max-model-len=4096

Canary rollout: deploy a second model to the same endpoint with --traffic-percentage 10; Vertex splits live traffic
between deployed models. (That is a traffic split, not request mirroring; mirroring is the gateway's shadow feature.)
"""
import argparse

from google.cloud import aiplatform


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--image", required=True, help="container image in Artifact Registry")
    ap.add_argument("--display-name", default="model-gateway")
    ap.add_argument("--endpoint", default="model-serving", help="endpoint display name (created if missing)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--predict-route", default="/predict")
    ap.add_argument("--health-route", default="/health")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--args", nargs=argparse.REMAINDER, default=None, help="container args (must be last)")
    ap.add_argument("--machine-type", default="g2-standard-8")
    ap.add_argument("--accelerator", default="NVIDIA_L4")
    ap.add_argument("--accelerator-count", type=int, default=1)
    ap.add_argument("--min-replicas", type=int, default=1)
    ap.add_argument("--max-replicas", type=int, default=5)
    ap.add_argument("--traffic-percentage", type=int, default=100)
    args = ap.parse_args()

    aiplatform.init(project=args.project, location=args.region)
    env = dict(kv.split("=", 1) for kv in args.env)

    model = aiplatform.Model.upload(
        display_name=args.display_name,
        serving_container_image_uri=args.image,
        serving_container_predict_route=args.predict_route,
        serving_container_health_route=args.health_route,
        serving_container_ports=[args.port],
        serving_container_environment_variables=env or None,
        serving_container_args=args.args or None,
    )

    existing = aiplatform.Endpoint.list(filter=f'display_name="{args.endpoint}"')
    endpoint = existing[0] if existing else aiplatform.Endpoint.create(display_name=args.endpoint)

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{args.display_name}-{model.version_id}",
        machine_type=args.machine_type,
        accelerator_type=args.accelerator,
        accelerator_count=args.accelerator_count,
        min_replica_count=args.min_replicas,
        max_replica_count=args.max_replicas,
        traffic_percentage=args.traffic_percentage,
    )
    print("endpoint:", endpoint.resource_name)


if __name__ == "__main__":
    main()
