#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/opt/venvs/mlops/bin/python"

NAMESPACE="${KFP_NAMESPACE:-kubeflow-user-example-com}"
HOST="${KFP_HOST:-http://localhost:8081/pipeline}"
EXPERIMENT="${KFP_EXPERIMENT:-electricity}"
FEATURES="${1:-lag_only}"
DATASET_KEY="datasets/LD2011_2014_kwh.parquet"

cleanup() {
  if [[ -n "${PF_PID:-}" ]] && kill -0 "$PF_PID" 2>/dev/null; then
    kill "$PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[1/6] Starting port-forward to istio ingress on 8081"
kubectl port-forward -n istio-system svc/istio-ingressgateway 8081:80 >/tmp/kf-port-forward.log 2>&1 &
PF_PID=$!
sleep 2

if ! kill -0 "$PF_PID" 2>/dev/null; then
  echo "Port-forward failed. Check /tmp/kf-port-forward.log"
  exit 1
fi

echo "[2/6] Generating ServiceAccount token"
export KF_TOKEN
KF_TOKEN="$(kubectl create token default-editor -n "$NAMESPACE" --audience=pipelines.kubeflow.org --duration=2h)"

echo "[3/6] Building local parquet if needed"
if [[ ! -f /tmp/LD2011_2014_kwh.parquet ]]; then
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd
source = Path('data/LD2011_2014.txt')
out = Path('/tmp/LD2011_2014_kwh.parquet')
if not source.exists():
    raise SystemExit('missing data/LD2011_2014.txt')
df = pd.read_csv(source, sep=';', decimal=',')
df = df.rename(columns={df.columns[0]: 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'])
for col in df.columns[1:]:
    df[col] = (df[col].astype('float32') / 4.0).astype('float32')
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(out)
PY
fi

echo "[4/6] Ensuring dataset exists in Garage S3"
if ! "$PYTHON_BIN" "$ROOT_DIR/garage_s3.py" ls models | grep -q "$DATASET_KEY"; then
  "$PYTHON_BIN" "$ROOT_DIR/garage_s3.py" upload /tmp/LD2011_2014_kwh.parquet --bucket models --key "$DATASET_KEY"
else
  echo "Dataset already present in s3://models/$DATASET_KEY"
fi

echo "[5/6] Compiling demo pipeline"
KFP_HOST="$HOST" KFP_NAMESPACE="$NAMESPACE" KFP_EXPERIMENT="$EXPERIMENT" \
  "$PYTHON_BIN" "$ROOT_DIR/cli_kfp_demo.py" compile

echo "[6/6] Submitting run (features=$FEATURES)"
KFP_HOST="$HOST" KFP_NAMESPACE="$NAMESPACE" KFP_EXPERIMENT="$EXPERIMENT" KF_TOKEN="$KF_TOKEN" \
  "$PYTHON_BIN" "$ROOT_DIR/cli_kfp_demo.py" run --features "$FEATURES"

echo "Demo done. Open: http://localhost:8081/pipeline"
