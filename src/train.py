import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.features import make_synthetic_dataset, LABELS

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse"))

    X, y = make_synthetic_dataset(n=4000, seed=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500)),
    ])

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("labels", ",".join(LABELS))
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))

        input_example = X_train[:5]
        signature = mlflow.models.infer_signature(input_example, pipe.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1(macro): {f1:.4f}")
        print(f"MODEL_URI: {model_uri}")

if __name__ == "__main__":
    main()
