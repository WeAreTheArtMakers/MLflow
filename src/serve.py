import json
import os
import base64
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.features import FEATURE_ORDER, LABELS, extract_image_features_from_bytes

app = FastAPI(title="ArtPulse API", version="0.3.0")


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


class PredictImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=16, description="Base64-encoded image content")


class PredictImageResponse(BaseModel):
    label: str
    prediction: int
    features: Dict[str, float]
    mlflow_model_uri: str


_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_model_uri: str = ""
_load_error: str = ""
_load_lock = Lock()


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def _registry_uri() -> str:
    return os.getenv("MLFLOW_REGISTRY_URI", "")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse")


def _model_name() -> str:
    return os.getenv("MODEL_NAME", "artpulse-classifier")


def _model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "champion")


def _use_registry_alias() -> bool:
    return os.getenv("USE_MODEL_REGISTRY_ALIAS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prediction_log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG_PATH", "./artifacts/prediction_events.jsonl"))


def _image_feature_size() -> int:
    return int(os.getenv("IMAGE_FEATURE_SIZE", "128"))


def _apply_mlflow_uris() -> None:
    mlflow.set_tracking_uri(_tracking_uri())
    if _registry_uri():
        mlflow.set_registry_uri(_registry_uri())


def _discover_model_uri() -> str:
    env_model_uri = os.getenv("MODEL_URI", "").strip()
    if env_model_uri:
        return env_model_uri

    if _use_registry_alias():
        return f"models:/{_model_name()}@{_model_alias()}"

    uri_file = Path(os.getenv("MODEL_URI_FILE", "./artifacts/latest_model_uri.txt"))
    if uri_file.exists():
        model_uri = uri_file.read_text(encoding="utf-8").strip()
        if model_uri:
            return model_uri

    client = MlflowClient(tracking_uri=_tracking_uri(), registry_uri=_registry_uri() or None)
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


def _append_prediction_events(
    rows: List[Dict[str, float]],
    prediction_ids: List[int],
    labels: List[str],
    source: str,
) -> None:
    if not rows:
        return

    log_path = _prediction_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        for row, pred_id, label in zip(rows, prediction_ids, labels):
            event = {
                "timestamp": now,
                "source": source,
                "model_uri": _model_uri,
                "features": row,
                "prediction": int(pred_id),
                "label": label,
            }
            fh.write(json.dumps(event) + "\n")


def load_model(force_reload: bool = False) -> None:
    global _model, _model_uri, _load_error
    with _load_lock:
        if _model is not None and not force_reload:
            return

        _apply_mlflow_uris()

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
        "mlflow_registry_uri": _registry_uri(),
        "mlflow_experiment": _experiment_name(),
        "mlflow_model_uri": _model_uri,
        "model_registry": {
            "enabled": _use_registry_alias(),
            "model_name": _model_name(),
            "model_alias": _model_alias(),
        },
        "prediction_log_path": str(_prediction_log_path()),
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


@app.post("/reload-model")
def reload_model() -> dict:
    try:
        load_model(force_reload=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
    return {"status": "reloaded", "mlflow_model_uri": _model_uri}


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
    X_np = np.asarray(rows, dtype=np.float32)
    preds = _model.predict(X_np)

    preds_int = [int(p) for p in np.asarray(preds).tolist()]
    labels = [LABELS[i] if 0 <= i < len(LABELS) else "unknown" for i in preds_int]

    row_dicts = [{name: float(getattr(row, name)) for name in FEATURE_ORDER} for row in req.rows]
    _append_prediction_events(row_dicts, preds_int, labels, source="tabular")

    return PredictResponse(labels=labels, predictions=preds_int, mlflow_model_uri=_model_uri)


@app.post("/predict-image", response_model=PredictImageResponse)
def predict_image(req: PredictImageRequest) -> PredictImageResponse:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    try:
        content = base64.b64decode(req.image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        feature_vector = extract_image_features_from_bytes(
            image_bytes=content,
            image_size=_image_feature_size(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image parsing failed: {exc}") from exc

    X_np = np.asarray([feature_vector], dtype=np.float32)
    preds = _model.predict(X_np)
    pred_id = int(np.asarray(preds).tolist()[0])
    label = LABELS[pred_id] if 0 <= pred_id < len(LABELS) else "unknown"

    feature_payload = {
        FEATURE_ORDER[idx]: float(feature_vector[idx]) for idx in range(len(FEATURE_ORDER))
    }
    _append_prediction_events([feature_payload], [pred_id], [label], source="image_upload")

    return PredictImageResponse(
        label=label,
        prediction=pred_id,
        features=feature_payload,
        mlflow_model_uri=_model_uri,
    )
