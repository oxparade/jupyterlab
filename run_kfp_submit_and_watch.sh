#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${KFP_ENV_FILE:-$SCRIPT_DIR/.env.kfp}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

NS="${KFP_NAMESPACE:-kubeflow-user-example-com}"
KFP_HOST="${KFP_HOST:-http://localhost:8081/pipeline}"
EXPERIMENT="${KFP_EXPERIMENT:-Default}"
PIPELINE_PACKAGE="${KFP_PIPELINE_PACKAGE:-kubeflow_pipeline.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venvs/mlops/bin/python}"

STRATEGY="${KFP_STRATEGY:-short_memory}"
SPLIT_STRATEGY="${KFP_SPLIT_STRATEGY:-recent_history}"
GOVERNANCE_DRY_RUN="${KFP_GOVERNANCE_DRY_RUN:-true}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow-server.mlflow.svc.cluster.local:5000}"

POLL_SECONDS="${KFP_POLL_SECONDS:-15}"
MAX_POLLS="${KFP_MAX_POLLS:-120}"

health_url="${KFP_HOST%/}/apis/v2beta1/healthz"
http_code="$(curl -sS -o /tmp/kfp_health_body.json -w '%{http_code}' "$health_url" || true)"
if [[ "$http_code" != "200" && "$http_code" != "401" ]]; then
  echo "KFP API not reachable on $health_url (HTTP $http_code)."
  echo "If needed, start port-forward: kubectl port-forward -n istio-system svc/istio-ingressgateway 8081:80"
  exit 1
fi

KF_TOKEN="$(kubectl create token default-editor -n "$NS" --audience=pipelines.kubeflow.org --duration=2h)"

submit_output="$($PYTHON_BIN submit_kubeflow_run.py \
  --host "$KFP_HOST" \
  --namespace "$NS" \
  --experiment-name "$EXPERIMENT" \
  --pipeline-package "$PIPELINE_PACKAGE" \
  --strategy "$STRATEGY" \
  --split-strategy "$SPLIT_STRATEGY" \
  --governance-dry-run "$GOVERNANCE_DRY_RUN" \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --token "$KF_TOKEN")"

echo "$submit_output"

run_id="$(printf '%s\n' "$submit_output" | grep -o '"run_id": "[^"]*"' | head -n1 | cut -d '"' -f4 || true)"
if [[ -n "$run_id" ]]; then
  WF="$(kubectl get wf -n "$NS" -l "pipeline/runid=$run_id" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')"
else
  WF="$(kubectl get wf -n "$NS" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')"
fi

echo "Monitoring $WF"
for _ in $(seq 1 "$MAX_POLLS"); do
  PHASE="$(kubectl get wf "$WF" -n "$NS" -o jsonpath='{.status.phase}')"
  TS="$(date +%H:%M:%S)"
  echo "$TS phase=$PHASE"
  if [[ "$PHASE" == "Succeeded" || "$PHASE" == "Failed" || "$PHASE" == "Error" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "== final =="
kubectl get wf "$WF" -n "$NS" -o wide
echo "== pods =="
kubectl get pods -n "$NS" -l workflows.argoproj.io/workflow="$WF"
