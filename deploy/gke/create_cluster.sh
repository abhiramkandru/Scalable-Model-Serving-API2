#!/usr/bin/env bash
# Provision GKE (CPU pool + L4 GPU pool), build and push images, deploy the stack.
# NOT TESTED against a live project: review every command, quota (GPUs!) and cost before running.
#
#   PROJECT_ID=my-proj REGION=us-central1 ./deploy/gke/create_cluster.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"          # pick a region with L4 capacity
CLUSTER="${CLUSTER:-model-serving}"
REPO="${REPO:-serving}"
TAG="${TAG:-1.0.0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

gcloud config set project "$PROJECT_ID"
gcloud services enable container.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com monitoring.googleapis.com aiplatform.googleapis.com

gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

# Regional cluster: CPU pool for the gateway/dashboard. Dataplane V2 enforces NetworkPolicy; managed Prometheus is on.
gcloud container clusters describe "$CLUSTER" --region "$REGION" >/dev/null 2>&1 || \
gcloud container clusters create "$CLUSTER" --region "$REGION" \
  --release-channel regular --enable-dataplane-v2 --enable-managed-prometheus \
  --workload-pool "${PROJECT_ID}.svc.id.goog" \
  --machine-type e2-standard-4 --num-nodes 1 --enable-autoscaling --min-nodes 1 --max-nodes 6

# GPU pool: one L4 per node, autoscaled; vllm.yaml selects it by cloud.google.com/gke-accelerator=nvidia-l4.
gcloud container node-pools describe gpu-l4 --cluster "$CLUSTER" --region "$REGION" >/dev/null 2>&1 || \
gcloud container node-pools create gpu-l4 --cluster "$CLUSTER" --region "$REGION" \
  --machine-type g2-standard-8 \
  --accelerator "type=nvidia-l4,count=1,gpu-driver-version=latest" \
  --enable-autoscaling --min-nodes 1 --max-nodes 8 --num-nodes 1

gcloud container clusters get-credentials "$CLUSTER" --region "$REGION"

# Build images in Cloud Build (no local Docker needed).
gcloud builds submit "$ROOT/server" --tag "${REGISTRY}/gateway:${TAG}"
gcloud builds submit "$ROOT/dashboard" --tag "${REGISTRY}/dashboard:${TAG}"

# Point kustomize at the pushed images, create the namespace + secret, apply.
cd "$ROOT/deploy/k8s"
kubectl apply -f namespace.yaml
kubectl -n model-serving get secret inference-api-keys >/dev/null 2>&1 || {
  echo "Create the API key secret first (see deploy/k8s/secrets.example.yaml), then re-run." >&2
  exit 1
}
kubectl kustomize . | \
  sed "s#REGION-docker.pkg.dev/PROJECT_ID/serving#${REGISTRY}#g" | kubectl apply -f -

echo "Deployed. Watch the model load:  kubectl -n model-serving logs -f deploy/vllm"
echo "Dashboard:                       kubectl -n model-serving port-forward svc/dashboard 8080:80"
