import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np

from src.features import FEATURE_ORDER, LABELS
from src.generate_image_dataset import generate_dataset
from src.train import train_models


def _to_feature_matrix(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURE_ORDER] for row in rows], dtype=np.float32)


def run_demo(
    tracking_uri: str,
    registry_uri: str,
    experiment_name: str,
    output_dir: str,
    run_train: bool,
    dataset_type: str,
    dataset_dir: str,
    n_samples: int,
    seed: int,
    test_size: float,
    register_best: bool,
    model_name: str,
    model_alias: str,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_uri_file = out_dir / "latest_model_uri.txt"
    summary_file = out_dir / "training_summary.json"

    training_summary: Dict[str, Any] = {}

    if run_train or not latest_uri_file.exists():
        training_summary = train_models(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri or None,
            experiment_name=experiment_name,
            dataset_type=dataset_type,
            dataset_dir=dataset_dir or None,
            n_samples=n_samples,
            seed=seed,
            test_size=test_size,
            output_dir=output_dir,
            register_best=register_best,
            model_name=model_name,
            model_alias=model_alias,
        )
    elif summary_file.exists():
        training_summary = json.loads(summary_file.read_text(encoding="utf-8"))

    model_uri = latest_uri_file.read_text(encoding="utf-8").strip()
    if not model_uri:
        raise RuntimeError("No model URI found. Run with --run-train first.")

    mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    sample_rows = [
        {
            "hue_mean": 0.10,
            "sat_mean": 0.90,
            "val_mean": 0.85,
            "contrast": 0.70,
            "edges": 0.35,
        },
        {
            "hue_mean": 0.55,
            "sat_mean": 0.15,
            "val_mean": 0.65,
            "contrast": 0.20,
            "edges": 0.10,
        },
    ]

    X = _to_feature_matrix(sample_rows)
    preds = model.predict(X)
    pred_ids = [int(p) for p in np.asarray(preds).tolist()]
    pred_labels = [LABELS[i] if 0 <= i < len(LABELS) else "unknown" for i in pred_ids]

    result = {
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "experiment_name": experiment_name,
        "model_uri": model_uri,
        "feature_order": FEATURE_ORDER,
        "sample_rows": sample_rows,
        "predictions": pred_ids,
        "labels": pred_labels,
        "best": training_summary.get("best", {}),
        "registry": training_summary.get("registry", {}),
    }

    (out_dir / "example_predictions.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end ArtPulse demo")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--registry-uri",
        default=os.getenv("MLFLOW_REGISTRY_URI", ""),
        help="MLflow registry URI",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse"),
        help="MLflow experiment name",
    )
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--run-train", action="store_true", help="Train candidates before prediction")
    parser.add_argument("--dataset-type", choices=["synthetic", "image"], default="synthetic")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--generate-sample-images", action="store_true")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--register-best", action="store_true")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "artpulse-classifier"))
    parser.add_argument("--model-alias", default=os.getenv("MODEL_ALIAS", "champion"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    if args.dataset_type == "image" and args.generate_sample_images:
        dataset_dir = dataset_dir or "data/images"
        generate_dataset(
            output_dir=dataset_dir,
            images_per_label=80,
            image_size=256,
            seed=args.seed,
        )
        print(f"Sample image dataset ready at: {Path(dataset_dir).resolve()}")

    result = run_demo(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        run_train=args.run_train,
        dataset_type=args.dataset_type,
        dataset_dir=dataset_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        test_size=args.test_size,
        register_best=args.register_best,
        model_name=args.model_name,
        model_alias=args.model_alias,
    )

    print("Demo tamamlandi.")
    print(f"Model URI: {result['model_uri']}")
    if result.get("best"):
        best = result["best"]
        print(
            "En iyi model: "
            f"{best.get('model_name', 'unknown')} "
            f"(f1_macro={best.get('f1_macro', 0):.4f}, accuracy={best.get('accuracy', 0):.4f})"
        )
    reg = result.get("registry", {})
    if reg:
        print(
            "Registry alias: "
            f"{reg.get('model_name', '-')}@{reg.get('model_alias', '-')} "
            f"(v{reg.get('model_version', '-')})"
        )
    print(f"Tahmin etiketleri: {result['labels']}")
    print("Cikti dosyasi: artifacts/example_predictions.json")


if __name__ == "__main__":
    main()
