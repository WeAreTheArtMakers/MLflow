import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _read_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("k6 summary payload must be a JSON object")
    return payload


def _metric_value(payload: Dict[str, Any], metric: str, field: str, default: float = 0.0) -> float:
    node = payload.get("metrics", {}).get(metric, {})
    values = node.get("values", {})
    val = values.get(field, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_markdown_report(summary: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    p95 = _metric_value(summary, "http_req_duration", "p(95)")
    p99 = _metric_value(summary, "http_req_duration", "p(99)")
    avg = _metric_value(summary, "http_req_duration", "avg")
    rps = _metric_value(summary, "http_reqs", "rate")
    fail_rate = _metric_value(summary, "http_req_failed", "rate")
    check_rate = _metric_value(summary, "checks", "rate")

    lines = [
        "# Load Test Report",
        "",
        f"Generated at (UTC): `{now}`",
        "",
        "## KPI Snapshot",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Request rate (req/s) | {rps:.2f} |",
        f"| Avg latency (ms) | {avg:.2f} |",
        f"| P95 latency (ms) | {p95:.2f} |",
        f"| P99 latency (ms) | {p99:.2f} |",
        f"| HTTP failure rate | {fail_rate:.4f} |",
        f"| Check pass rate | {check_rate:.4f} |",
        "",
        "## Interpretation",
        "",
        "- Compare P95/P99 against your SLA targets in `README.md`.",
        "- Keep failure rate below `2%` for baseline production readiness.",
        "- Re-run this test after any rollout change (canary/blue-green percentage).",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown load test report from k6 summary JSON")
    parser.add_argument("--input", required=True, help="Path to k6 summary JSON")
    parser.add_argument("--output", default="artifacts/loadtest_report.md", help="Output markdown path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    summary = _read_summary(input_path)
    markdown = build_markdown_report(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Load test report written: {output_path}")


if __name__ == "__main__":
    main()
