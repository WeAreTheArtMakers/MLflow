import os
from pathlib import Path
from threading import Lock
from typing import List, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.features import FEATURE_ORDER, LABELS

app = FastAPI(title="ArtPulse API", version="0.2.0")


class FeatureRow(BaseModel):
    hue_mean: float = Field(..., ge=0.0, le=1.0)
    sat_mean: float = Field(..., ge=0.0, le=1.0)
    val_mean: float = Field(..., ge=0.0, le=1.0)
    contrast: float = Field(..., ge=0.0, le=1.0)
    edges: float = Field(..., ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    rows: List[FeatureRow] = Field(..., min_length=1, description="Feature rows")


class PredictResponse(BaseModel):
    labels: List[str]
    predictions: List[int]
    mlflow_model_uri: str


_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_model_uri: str = ""
_load_error: str = ""
_load_lock = Lock()


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse")


def _discover_model_uri() -> str:
    env_model_uri = os.getenv("MODEL_URI", "").strip()
    if env_model_uri:
        return env_model_uri

    uri_file = Path(os.getenv("MODEL_URI_FILE", "./artifacts/latest_model_uri.txt"))
    if uri_file.exists():
        model_uri = uri_file.read_text(encoding="utf-8").strip()
        if model_uri:
            return model_uri

    client = MlflowClient(tracking_uri=_tracking_uri())
    exp = client.get_experiment_by_name(_experiment_name())
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{_experiment_name()}' not found. Train first with python3 -m src.train"
        )

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string='tags.deployment_ready = "true"',
        order_by=["metrics.f1_macro DESC", "attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["metrics.f1_macro DESC", "attributes.start_time DESC"],
            max_results=1,
        )
    if not runs:
        raise RuntimeError(
            f"No MLflow runs found in experiment '{_experiment_name()}'. Train first with python3 -m src.train"
        )

    return f"runs:/{runs[0].info.run_id}/model"


def load_model(force_reload: bool = False) -> None:
    global _model, _model_uri, _load_error
    with _load_lock:
        if _model is not None and not force_reload:
            return

        mlflow.set_tracking_uri(_tracking_uri())

        model_uri = _discover_model_uri()
        loaded = mlflow.pyfunc.load_model(model_uri)

        _model = loaded
        _model_uri = model_uri
        _load_error = ""


@app.on_event("startup")
def on_startup() -> None:
    try:
        load_model()
    except Exception as exc:
        global _load_error
        _load_error = str(exc)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "mlflow_tracking_uri": _tracking_uri(),
        "mlflow_experiment": _experiment_name(),
        "mlflow_model_uri": _model_uri,
        "error": _load_error,
    }


@app.get("/ready")
def ready() -> dict:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc
    return {"status": "ready", "mlflow_model_uri": _model_uri}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "labels": LABELS,
        "feature_order": FEATURE_ORDER,
        "mlflow_model_uri": _model_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    rows = [[getattr(row, name) for name in FEATURE_ORDER] for row in req.rows]
    X_np = np.asarray(rows, dtype=float)
    preds = _model.predict(X_np)

    preds_int = [int(p) for p in np.asarray(preds).tolist()]
    labels = [LABELS[i] if 0 <= i < len(LABELS) else "unknown" for i in preds_int]

    return PredictResponse(labels=labels, predictions=preds_int, mlflow_model_uri=_model_uri)
