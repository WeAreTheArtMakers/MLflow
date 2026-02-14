import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.features import FEATURE_ORDER


def _load_baseline_stats(training_summary_path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(training_summary_path.read_text(encoding="utf-8"))
    stats = payload.get("baseline_feature_stats", {})
    if not stats:
        raise ValueError(
            f"baseline_feature_stats not found in {training_summary_path}. Train with updated src.train first."
        )
    return stats


def _read_recent_events(prediction_log_path: Path, window_size: int) -> List[Dict[str, Any]]:
    if not prediction_log_path.exists():
        return []

    lines = prediction_log_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []

    selected = lines[-window_size:]
    events = []
    for line in selected:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _compute_recent_feature_stats(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = {name: [] for name in FEATURE_ORDER}
    for event in events:
        features = event.get("features", {})
        for name in FEATURE_ORDER:
            if name in features:
                values[name].append(float(features[name]))

    stats: Dict[str, Dict[str, float]] = {}
    for name in FEATURE_ORDER:
        arr = np.asarray(values[name], dtype=float)
        if arr.size == 0:
            stats[name] = {"mean": 0.0, "std": 0.0, "count": 0}
            continue
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "count": int(arr.size),
        }
    return stats


def build_drift_report(
    baseline_stats: Dict[str, Dict[str, float]],
    recent_stats: Dict[str, Dict[str, float]],
    z_threshold: float,
    event_count: int,
) -> Dict[str, Any]:
    feature_drift: Dict[str, Dict[str, float | bool]] = {}
    max_z = 0.0

    for name in FEATURE_ORDER:
        baseline_mean = float(baseline_stats[name]["mean"])
        baseline_std = max(float(baseline_stats[name]["std"]), 1e-6)
        recent_mean = float(recent_stats[name]["mean"])
        recent_std = float(recent_stats[name]["std"])

        z_score = abs(recent_mean - baseline_mean) / baseline_std
        max_z = max(max_z, z_score)

        feature_drift[name] = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "recent_mean": recent_mean,
            "recent_std": recent_std,
            "z_score_mean_shift": float(z_score),
            "drifted": bool(z_score >= z_threshold),
        }

    drift_detected = any(item["drifted"] for item in feature_drift.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": event_count,
        "z_threshold": z_threshold,
        "drift_detected": drift_detected,
        "drift_score": float(max_z),
        "feature_drift": feature_drift,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor feature drift from prediction logs")
    parser.add_argument("--training-summary", default="artifacts/training_summary.json")
    parser.add_argument("--prediction-log", default="artifacts/prediction_events.jsonl")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--z-threshold", type=float, default=2.5)
    parser.add_argument("--output", default="artifacts/drift_report.json")
    parser.add_argument("--history-output", default="artifacts/drift_history.jsonl")
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_summary_path = Path(args.training_summary)
    prediction_log_path = Path(args.prediction_log)
    output_path = Path(args.output)
    history_path = Path(args.history_output)

    baseline_stats = _load_baseline_stats(training_summary_path)
    events = _read_recent_events(prediction_log_path, window_size=args.window_size)

    if not events:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "event_count": 0,
            "status": "no_events",
            "message": "No prediction events found yet.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    recent_stats = _compute_recent_feature_stats(events)
    report = build_drift_report(
        baseline_stats=baseline_stats,
        recent_stats=recent_stats,
        z_threshold=args.z_threshold,
        event_count=len(events),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))

    if args.fail_on_drift and report["drift_detected"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
