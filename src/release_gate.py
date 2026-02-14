import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _read_json(url: str, headers: Dict[str, str] | None = None, timeout: float = 15.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _prom_query(prometheus_url: str, expr: str) -> float:
    endpoint = prometheus_url.rstrip("/") + "/api/v1/query"
    query = urllib.parse.urlencode({"query": expr})
    payload = _read_json(f"{endpoint}?{query}")
    if payload.get("status") != "success":
        raise ValueError(f"Prometheus query failed: {payload}")
    data = payload.get("data", {})
    result = data.get("result", [])
    if not isinstance(result, list) or not result:
        return 0.0
    value = result[0].get("value", [None, "0"])
    number = float(value[1])
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def _ops_summary(api_url: str, api_key: str) -> Dict[str, Any]:
    headers = {"x-api-key": api_key}
    return _read_json(api_url.rstrip("/") + "/ops/summary", headers=headers)


def evaluate_release_gate(
    api_url: str,
    api_key: str,
    prometheus_url: str,
    p95_ms_limit: float,
    error_rate_limit: float,
    drift_score_limit: float,
) -> Dict[str, Any]:
    p95_expr = (
        'histogram_quantile(0.95, sum(rate(artpulse_http_request_latency_seconds_bucket'
        '{path=~"/predict|/predict-image|/demo/predict"}[5m])) by (le))'
    )
    err_expr = (
        'sum(rate(artpulse_http_requests_total{status=~"5.."}[5m])) '
        '/ clamp_min(sum(rate(artpulse_http_requests_total[5m])), 1)'
    )

    p95_seconds = _prom_query(prometheus_url, p95_expr)
    error_rate = _prom_query(prometheus_url, err_expr)
    ops = _ops_summary(api_url, api_key=api_key)
    drift_score = float(ops.get("drift", {}).get("drift_score", 0.0))

    checks = {
        "p95_ms": {
            "value": p95_seconds * 1000.0,
            "limit": p95_ms_limit,
            "passed": (p95_seconds * 1000.0) <= p95_ms_limit,
        },
        "error_rate": {
            "value": error_rate,
            "limit": error_rate_limit,
            "passed": error_rate <= error_rate_limit,
        },
        "drift_score": {
            "value": drift_score,
            "limit": drift_score_limit,
            "passed": drift_score <= drift_score_limit,
        },
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "prometheus_url": prometheus_url,
        "passed": passed,
        "checks": checks,
        "rollout": ops.get("rollout", {}),
        "model": ops.get("model", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release quality gate for canary/production promotion")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument("--p95-ms", type=float, default=180.0)
    parser.add_argument("--error-rate-max", type=float, default=0.01)
    parser.add_argument("--drift-score-max", type=float, default=2.5)
    parser.add_argument("--output", default="artifacts/release_gate_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_release_gate(
        api_url=args.api_url,
        api_key=args.api_key,
        prometheus_url=args.prometheus_url,
        p95_ms_limit=args.p95_ms,
        error_rate_limit=args.error_rate_max,
        drift_score_limit=args.drift_score_max,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not report["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
