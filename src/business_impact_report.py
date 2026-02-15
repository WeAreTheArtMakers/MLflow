import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _fmt_number(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _extract_accuracy(training_summary: Optional[Dict[str, Any]]) -> Optional[float]:
    if not training_summary:
        return None
    best = training_summary.get("best", {})
    return _to_float(best.get("accuracy"))


def _extract_f1(training_summary: Optional[Dict[str, Any]]) -> Optional[float]:
    if not training_summary:
        return None
    best = training_summary.get("best", {})
    return _to_float(best.get("f1_macro"))


def _extract_p95_ms(
    loadtest_summary: Optional[Dict[str, Any]], release_gate_report: Optional[Dict[str, Any]]
) -> Optional[float]:
    if loadtest_summary:
        p95 = (
            loadtest_summary.get("metrics", {})
            .get("http_req_duration", {})
            .get("values", {})
            .get("p(95)")
        )
        val = _to_float(p95)
        if val is not None:
            return val

    if release_gate_report:
        val = (
            release_gate_report.get("checks", {})
            .get("p95_ms", {})
            .get("value")
        )
        return _to_float(val)
    return None


def _extract_error_rate(
    loadtest_summary: Optional[Dict[str, Any]], release_gate_report: Optional[Dict[str, Any]]
) -> Optional[float]:
    if loadtest_summary:
        fail_rate = (
            loadtest_summary.get("metrics", {})
            .get("http_req_failed", {})
            .get("values", {})
            .get("rate")
        )
        val = _to_float(fail_rate)
        if val is not None:
            return val

    if release_gate_report:
        val = (
            release_gate_report.get("checks", {})
            .get("error_rate", {})
            .get("value")
        )
        return _to_float(val)
    return None


def _extract_drift_score(
    drift_report: Optional[Dict[str, Any]], release_gate_report: Optional[Dict[str, Any]]
) -> Optional[float]:
    if drift_report:
        val = _to_float(drift_report.get("drift_score"))
        if val is not None:
            return val
    if release_gate_report:
        val = (
            release_gate_report.get("checks", {})
            .get("drift_score", {})
            .get("value")
        )
        return _to_float(val)
    return None


def _kpi_status(value: Optional[float], limit: Optional[float], lower_is_better: bool) -> str:
    if value is None or limit is None:
        return "n/a"
    if lower_is_better:
        return "pass" if value <= limit else "fail"
    return "pass" if value >= limit else "fail"


def build_report(
    environment: str,
    api_base_url: str,
    training_summary: Optional[Dict[str, Any]],
    drift_report: Optional[Dict[str, Any]],
    loadtest_summary: Optional[Dict[str, Any]],
    release_gate_report: Optional[Dict[str, Any]],
    deploy_lead_time_min: Optional[float],
    rollback_time_min: Optional[float],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    accuracy = _extract_accuracy(training_summary)
    f1_macro = _extract_f1(training_summary)
    p95_ms = _extract_p95_ms(loadtest_summary, release_gate_report)
    error_rate = _extract_error_rate(loadtest_summary, release_gate_report)
    drift_score = _extract_drift_score(drift_report, release_gate_report)

    lines = [
        "# Business Impact One-Pager",
        "",
        f"- Generated at (UTC): `{generated_at}`",
        f"- Environment: `{environment}`",
        f"- API base URL: `{api_base_url}`",
        "",
        "## KPI Snapshot",
        "",
        "| KPI | Current | Target | Status | Source |",
        "| --- | ---: | ---: | --- | --- |",
        f"| Accuracy | {_fmt_number(accuracy, 4)} | >= 0.92 | {_kpi_status(accuracy, 0.92, lower_is_better=False)} | training_summary.json |",
        f"| Macro F1 | {_fmt_number(f1_macro, 4)} | >= 0.90 | {_kpi_status(f1_macro, 0.90, lower_is_better=False)} | training_summary.json |",
        f"| P95 latency (ms) | {_fmt_number(p95_ms, 2)} | <= 180.00 | {_kpi_status(p95_ms, 180.0, lower_is_better=True)} | k6/release_gate |",
        f"| Error rate | {_fmt_number(error_rate, 4)} | <= 0.0100 | {_kpi_status(error_rate, 0.01, lower_is_better=True)} | k6/release_gate |",
        f"| Drift score | {_fmt_number(drift_score, 4)} | <= 2.5000 | {_kpi_status(drift_score, 2.5, lower_is_better=True)} | drift_report/release_gate |",
        f"| Deploy lead time (min) | {_fmt_number(deploy_lead_time_min, 2)} | <= 15.00 | {_kpi_status(deploy_lead_time_min, 15.0, lower_is_better=True)} | deployment workflow |",
        f"| Rollback time (min) | {_fmt_number(rollback_time_min, 2)} | <= 3.00 | {_kpi_status(rollback_time_min, 3.0, lower_is_better=True)} | rollback drill |",
        "",
        "## Operational Notes",
        "",
        "- Use staging as mandatory gate before production promotion.",
        "- Keep `run_quality_gate=true` for every production deployment.",
        "- Run weekly rollback drill with forced gate-fail threshold in staging.",
        "",
    ]
    return "\n".join(lines)


def _optional_float(raw: str) -> Optional[float]:
    value = raw.strip()
    if not value:
        return None
    return _to_float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one-page business impact report from ArtPulse artifacts")
    parser.add_argument("--training-summary", default="artifacts/training_summary.json")
    parser.add_argument("--drift-report", default="artifacts/drift_report.json")
    parser.add_argument("--loadtest-summary", default="loadtest/k6/k6-summary.json")
    parser.add_argument("--release-gate-report", default="artifacts/release_gate_report.json")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--api-base-url", default="https://api.wearetheartmakers.com")
    parser.add_argument("--deploy-lead-time-min", default="")
    parser.add_argument("--rollback-time-min", default="")
    parser.add_argument("--output", default="artifacts/business_impact_onepager.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        environment=args.environment,
        api_base_url=args.api_base_url,
        training_summary=_load_json(Path(args.training_summary)),
        drift_report=_load_json(Path(args.drift_report)),
        loadtest_summary=_load_json(Path(args.loadtest_summary)),
        release_gate_report=_load_json(Path(args.release_gate_report)),
        deploy_lead_time_min=_optional_float(args.deploy_lead_time_min),
        rollback_time_min=_optional_float(args.rollback_time_min),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Business impact report written: {output_path}")


if __name__ == "__main__":
    main()
