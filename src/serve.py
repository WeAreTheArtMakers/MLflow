import os
from typing import List, Dict, Optional

import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import LABELS

app = FastAPI(title="ArtPulse API", version="0.1.0")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MODEL_URI_ENV = os.getenv("MODEL_URI", "")

_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_mlflow_model_uri: str = ""

FEATURE_ORDER = ["hue_mean", "sat_mean", "val_mean", "contrast", "edges"]

class PredictRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(..., description="List of feature dicts.")

class PredictResponse(BaseModel):
    labels: List[str]
    predictions: List[int]
    mlflow_model_uri: str

def load_model() -> None:
    global _model, _model_uri
    if _model is not None:
        return

    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = MODEL_URI_ENV
    if not model_uri:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exp = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse"))
        if exp is None:
            raise RuntimeError("No experiment found. Train first: python3 -m src.train")
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
        if not runs:
            raise RuntimeError("No runs found. Train first: python3 -m src.train")
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"

    _model = mlflow.pyfunc.load_model(model_uri)
    _model_uri = model_uri

@app.on_event("startup")
def on_startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "mlflow_model_uri": _model_uri}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        load_model()

    X = []
    for row in req.rows:
        X.append([float(row.get(k, 0.0)) for k in FEATURE_ORDER])

    X_np = np.array(X, dtype=float)
    preds = _model.predict(X_np)
    preds_int = [int(p) for p in np.asarray(preds).tolist()]
    labels = [LABELS[i] if 0 <= i < len(LABELS) else "unknown" for i in preds_int]

    return PredictResponse(labels=labels, predictions=preds_int, mlflow_model_uri=_model_uri)
