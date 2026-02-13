import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from src.train import train_models


def _should_retrain(only_if_drift: bool, drift_report_path: Path) -> bool:
    if not only_if_drift:
        return True
    if not drift_report_path.exists():
        return False
    report = json.loads(drift_report_path.read_text(encoding="utf-8"))
    return bool(report.get("drift_detected", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic retraining job")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )
    parser.add_argument(
        "--registry-uri",
        default=os.getenv("MLFLOW_REGISTRY_URI", ""),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse"),
    )
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", ""))
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--register-best", action="store_true")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "artpulse-classifier"))
    parser.add_argument("--model-alias", default=os.getenv("MODEL_ALIAS", "challenger"))
    parser.add_argument("--only-if-drift", action="store_true")
    parser.add_argument("--drift-report", default="artifacts/drift_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not _should_retrain(args.only_if_drift, Path(args.drift_report)):
        print("Skipping retraining: drift not detected.")
        return

    dataset_dir = args.dataset_dir.strip()
    dataset_type = "image" if dataset_dir and Path(dataset_dir).exists() else "synthetic"

    summary: Dict[str, Any] = train_models(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri or None,
        experiment_name=args.experiment_name,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir or None,
        n_samples=args.n_samples,
        seed=args.seed,
        test_size=args.test_size,
        output_dir=args.output_dir,
        register_best=args.register_best,
        model_name=args.model_name,
        model_alias=args.model_alias,
    )

    print(f"Retraining completed. Best model URI: {summary['best_model_uri']}")


if __name__ == "__main__":
    main()
