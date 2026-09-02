"""Module 3 Kubeflow smoke checks for the TP environment.

Checks (slides 1-12 scope):
- kubectl access
- expected namespaces
- control-plane and user-namespace pods
- Istio ingress service availability
- ServiceAccount token generation (audience: pipelines.kubeflow.org)
- KFP pipeline compilation
- KFP API reachability via ingress port-forward

Run:
    /opt/venvs/mlops/bin/python verify_kubeflow_module3.py
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

USER_NS = "kubeflow-user-example-com"
KFP_AUDIENCE = "pipelines.kubeflow.org"


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _check_kubectl() -> CheckResult:
    cp = _run(["kubectl", "version", "--client"])
    return CheckResult(
        name="kubectl_client",
        ok=cp.returncode == 0,
        details=(cp.stdout or cp.stderr).strip(),
    )


def _check_namespaces() -> CheckResult:
    cp = _run(["kubectl", "get", "ns", "-o", "name"])
    if cp.returncode != 0:
        return CheckResult("namespaces", False, (cp.stderr or cp.stdout).strip())
    lines = cp.stdout.splitlines()
    required = ["namespace/kubeflow", f"namespace/{USER_NS}"]
    missing = [ns for ns in required if ns not in lines]
    return CheckResult(
        name="namespaces",
        ok=not missing,
        details=("missing=" + ",".join(missing)) if missing else "kubeflow + user ns present",
    )


def _check_pods(namespace: str, name: str) -> CheckResult:
    cp = _run(["kubectl", "get", "pods", "-n", namespace, "--no-headers"])
    if cp.returncode != 0:
        return CheckResult(name, False, (cp.stderr or cp.stdout).strip())

    rows = [line for line in cp.stdout.splitlines() if line.strip()]
    if not rows:
        if namespace == USER_NS:
            return CheckResult(name, True, "no pods yet (expected before first run)")
        return CheckResult(name, False, "no pods")

    running = 0
    not_running: list[str] = []
    for row in rows:
        parts = row.split()
        pod_name = parts[0]
        phase = parts[2] if len(parts) >= 3 else "?"
        if phase == "Running" or phase == "Completed":
            running += 1
        else:
            not_running.append(f"{pod_name}:{phase}")

    ok = len(not_running) == 0
    details = f"pods={len(rows)} running_or_completed={running}"
    if not_running:
        details += " not_running=" + ";".join(not_running[:5])
    return CheckResult(name, ok, details)


def _check_istio_ingress() -> CheckResult:
    cp = _run(["kubectl", "get", "svc", "istio-ingressgateway", "-n", "istio-system"])
    return CheckResult(
        name="istio_ingressgateway",
        ok=cp.returncode == 0,
        details=(cp.stdout or cp.stderr).strip().splitlines()[0] if (cp.stdout or cp.stderr) else "",
    )


def _check_token() -> CheckResult:
    cp = _run(
        [
            "kubectl",
            "create",
            "token",
            "default-editor",
            "-n",
            USER_NS,
            "--audience",
            KFP_AUDIENCE,
            "--duration=10m",
        ]
    )
    if cp.returncode != 0:
        return CheckResult("kfp_token", False, (cp.stderr or cp.stdout).strip())
    token = cp.stdout.strip()
    return CheckResult("kfp_token", len(token) > 40, f"token_len={len(token)}")


def _check_compile() -> CheckResult:
    pipeline_file = Path("kubeflow_pipeline.py")
    if not pipeline_file.exists():
        return CheckResult("kfp_compile", False, "kubeflow_pipeline.py missing")

    out = Path("/tmp/module3_kubeflow_pipeline.yaml")
    cp = _run(["/opt/venvs/mlops/bin/python", str(pipeline_file), "--output", str(out)], timeout=120)
    if cp.returncode != 0:
        return CheckResult("kfp_compile", False, (cp.stderr or cp.stdout).strip())
    if not out.exists():
        return CheckResult("kfp_compile", False, "compiled file not found")
    return CheckResult("kfp_compile", True, f"compiled={out} size={out.stat().st_size}")


def _check_kfp_health_via_ingress() -> CheckResult:
    pf = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "-n",
            "istio-system",
            "svc/istio-ingressgateway",
            "8081:80",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)
        cp = _run(["curl", "-sS", "-o", "/tmp/kfp_health_body.json", "-w", "%{http_code}", "http://127.0.0.1:8081/pipeline/apis/v2beta1/healthz"], timeout=20)
        if cp.returncode != 0:
            return CheckResult("kfp_health_ingress", False, (cp.stderr or cp.stdout).strip())
        code = cp.stdout.strip()
        ok = code.startswith("2") or code.startswith("3") or code == "401"
        body = Path("/tmp/kfp_health_body.json").read_text(encoding="utf-8", errors="ignore")
        body_sample = body[:180].replace("\n", " ")
        return CheckResult("kfp_health_ingress", ok, f"http={code} body={body_sample}")
    finally:
        pf.terminate()


def main() -> None:
    checks = [
        _check_kubectl(),
        _check_namespaces(),
        _check_pods("kubeflow", "kubeflow_pods"),
        _check_pods(USER_NS, "user_ns_pods"),
        _check_istio_ingress(),
        _check_token(),
        _check_compile(),
        _check_kfp_health_via_ingress(),
    ]

    summary = {
        "all_ok": all(c.ok for c in checks),
        "ok": sum(1 for c in checks if c.ok),
        "total": len(checks),
        "checks": [asdict(c) for c in checks],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not summary["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
