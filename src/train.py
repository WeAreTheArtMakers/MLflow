import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_ORDER, LABELS, make_synthetic_dataset


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


def train_models(
    tracking_uri: str,
    experiment_name: str,
    n_samples: int,
    seed: int,
    test_size: float,
    output_dir: str,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X, y = make_synthetic_dataset(n=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    run_summaries: List[TrainingRunSummary] = []

    for model_name, model in build_model_candidates(seed):
        with mlflow.start_run(run_name=f"{model_name}-seed-{seed}") as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average="macro"))

            mlflow.set_tag("project", "artpulse")
            mlflow.set_tag("task", "classification")
            mlflow.set_tag("candidate_model", model_name)

            mlflow.log_param("model", model_name)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("feature_order", ",".join(FEATURE_ORDER))
            mlflow.log_param("labels", ",".join(LABELS))
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)

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
                    model_name=model_name,
                    run_id=run.info.run_id,
                    accuracy=acc,
                    f1_macro=f1,
                    model_uri=model_uri,
                )
            )

    best = max(run_summaries, key=lambda x: (x.f1_macro, x.accuracy))
    client = MlflowClient(tracking_uri=tracking_uri)
    for summary in run_summaries:
        client.set_tag(
            summary.run_id,
            "deployment_ready",
            "true" if summary.run_id == best.run_id else "false",
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "latest_model_uri.txt").write_text(f"{best.model_uri}\n", encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "best": asdict(best),
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
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse"),
        help="MLflow experiment name",
    )
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_models(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        n_samples=args.n_samples,
        seed=args.seed,
        test_size=args.test_size,
        output_dir=args.output_dir,
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
    print(f"BEST_MODEL_URI: {best['model_uri']}")


if __name__ == "__main__":
    main()
