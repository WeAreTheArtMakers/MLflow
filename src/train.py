import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    FEATURE_ORDER,
    LABELS,
    build_image_feature_dataset,
    make_synthetic_dataset,
)


@dataclass
class TrainingRunSummary:
    model_name: str
    run_id: str
    accuracy: float
    f1_macro: float
    model_uri: str


def build_model_candidates(seed: int) -> List[Tuple[str, object]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.08,
                max_iter=250,
                random_state=seed,
            ),
        ),
    ]


def _apply_mlflow_uris(tracking_uri: str, registry_uri: Optional[str] = None) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)


def _compute_feature_stats(X: Any) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for idx, feature_name in enumerate(FEATURE_ORDER):
        col = X[:, idx]
        stats[feature_name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
        }
    return stats


def _load_training_dataset(
    dataset_type: str,
    n_samples: int,
    seed: int,
    dataset_dir: Optional[str],
    image_size: int,
    min_images_per_label: int,
) -> Tuple[Any, Any, Dict[str, Any]]:
    if dataset_type == "synthetic":
        X, y = make_synthetic_dataset(n=n_samples, seed=seed)
        metadata: Dict[str, Any] = {
            "dataset_type": "synthetic",
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "labels": LABELS,
        }
        return X, y, metadata

    if not dataset_dir:
        raise ValueError("dataset_dir is required when dataset_type=image")

    X, y, image_meta = build_image_feature_dataset(
        dataset_dir=dataset_dir,
        labels=LABELS,
        image_size=image_size,
        min_images_per_label=min_images_per_label,
        strict_labels=False,
    )
    metadata = {
        "dataset_type": "image",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "labels": LABELS,
        "image_meta": image_meta,
    }
    return X, y, metadata


def _register_best_model(
    best_model_uri: str,
    tracking_uri: str,
    registry_uri: Optional[str],
    model_name: str,
    model_alias: str,
) -> Dict[str, Any]:
    _apply_mlflow_uris(tracking_uri=tracking_uri, registry_uri=registry_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    model_version = mlflow.register_model(model_uri=best_model_uri, name=model_name)

    for _ in range(30):
        mv = client.get_model_version(name=model_name, version=model_version.version)
        if mv.status == "READY":
            break
        time.sleep(2)

    client.set_registered_model_alias(name=model_name, alias=model_alias, version=model_version.version)

    return {
        "model_name": model_name,
        "model_version": str(model_version.version),
        "model_alias": model_alias,
        "alias_uri": f"models:/{model_name}@{model_alias}",
    }


def train_models(
    tracking_uri: str,
    experiment_name: str,
    n_samples: int,
    seed: int,
    test_size: float,
    output_dir: str,
    dataset_type: str = "synthetic",
    dataset_dir: Optional[str] = None,
    image_size: int = 128,
    min_images_per_label: int = 5,
    register_best: bool = False,
    model_name: str = "artpulse-classifier",
    model_alias: str = "champion",
    registry_uri: Optional[str] = None,
) -> Dict[str, Any]:
    _apply_mlflow_uris(tracking_uri=tracking_uri, registry_uri=registry_uri)
    mlflow.set_experiment(experiment_name)

    X, y, dataset_meta = _load_training_dataset(
        dataset_type=dataset_type,
        n_samples=n_samples,
        seed=seed,
        dataset_dir=dataset_dir,
        image_size=image_size,
        min_images_per_label=min_images_per_label,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_feature_stats = _compute_feature_stats(X_train)

    run_summaries: List[TrainingRunSummary] = []

    for model_name_candidate, model in build_model_candidates(seed):
        with mlflow.start_run(run_name=f"{model_name_candidate}-seed-{seed}") as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average="macro"))

            mlflow.set_tag("project", "artpulse")
            mlflow.set_tag("task", "classification")
            mlflow.set_tag("candidate_model", model_name_candidate)

            mlflow.log_param("model", model_name_candidate)
            mlflow.log_param("seed", seed)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("dataset_type", dataset_type)
            mlflow.log_param("feature_order", ",".join(FEATURE_ORDER))
            mlflow.log_param("labels", ",".join(LABELS))
            if dataset_type == "synthetic":
                mlflow.log_param("n_samples", n_samples)
            else:
                mlflow.log_param("dataset_dir", dataset_meta["image_meta"]["dataset_dir"])
                mlflow.log_param("image_size", image_size)
                mlflow.log_param("min_images_per_label", min_images_per_label)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_dict(dataset_meta, "dataset_meta.json")
            mlflow.log_dict(train_feature_stats, "baseline_feature_stats.json")

            input_example = X_train[:5]
            signature = mlflow.models.infer_signature(
                input_example, model.predict(input_example)
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
            )

            model_uri = f"runs:/{run.info.run_id}/model"
            run_summaries.append(
                TrainingRunSummary(
                    model_name=model_name_candidate,
                    run_id=run.info.run_id,
                    accuracy=acc,
                    f1_macro=f1,
                    model_uri=model_uri,
                )
            )

    best = max(run_summaries, key=lambda x: (x.f1_macro, x.accuracy))
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    for summary in run_summaries:
        client.set_tag(
            summary.run_id,
            "deployment_ready",
            "true" if summary.run_id == best.run_id else "false",
        )

    registry_result: Dict[str, Any] = {}
    preferred_model_uri = best.model_uri
    if register_best:
        registry_result = _register_best_model(
            best_model_uri=best.model_uri,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            model_name=model_name,
            model_alias=model_alias,
        )
        preferred_model_uri = registry_result["alias_uri"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "latest_model_uri.txt").write_text(f"{preferred_model_uri}\n", encoding="utf-8")
    (out_dir / "baseline_feature_stats.json").write_text(
        json.dumps(train_feature_stats, indent=2), encoding="utf-8"
    )

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
        "experiment_name": experiment_name,
        "dataset": dataset_meta,
        "feature_order": FEATURE_ORDER,
        "baseline_feature_stats": train_feature_stats,
        "best": asdict(best),
        "best_model_uri": preferred_model_uri,
        "registry": registry_result,
        "candidates": [asdict(s) for s in run_summaries],
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and track ArtPulse models with MLflow")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--registry-uri",
        default=os.getenv("MLFLOW_REGISTRY_URI", ""),
        help="MLflow registry URI for Model Registry",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse"),
        help="MLflow experiment name",
    )
    parser.add_argument("--dataset-type", choices=["synthetic", "image"], default="synthetic")
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", ""))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--min-images-per-label", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--register-best", action="store_true")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "artpulse-classifier"))
    parser.add_argument("--model-alias", default=os.getenv("MODEL_ALIAS", "champion"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_models(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri or None,
        experiment_name=args.experiment_name,
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir or None,
        image_size=args.image_size,
        min_images_per_label=args.min_images_per_label,
        n_samples=args.n_samples,
        seed=args.seed,
        test_size=args.test_size,
        output_dir=args.output_dir,
        register_best=args.register_best,
        model_name=args.model_name,
        model_alias=args.model_alias,
    )

    best = summary["best"]
    print("Model leaderboard:")
    for cand in sorted(summary["candidates"], key=lambda x: x["f1_macro"], reverse=True):
        print(
            f"- {cand['model_name']:<24} f1_macro={cand['f1_macro']:.4f} "
            f"accuracy={cand['accuracy']:.4f} run_id={cand['run_id']}"
        )
    print(f"BEST_RUN_ID: {best['run_id']}")
    print(f"BEST_MODEL: {best['model_name']}")
    print(f"BEST_MODEL_URI: {summary['best_model_uri']}")
    if summary["registry"]:
        reg = summary["registry"]
        print(
            "MODEL_REGISTRY: "
            f"name={reg['model_name']} version={reg['model_version']} alias={reg['model_alias']}"
        )


if __name__ == "__main__":
    main()
