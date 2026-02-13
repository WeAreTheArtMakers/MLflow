import importlib
import os
from pathlib import Path
from io import BytesIO
import base64

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.train import train_models


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("artpulse")
    tracking_dir = tmp_dir / "mlruns"
    output_dir = tmp_dir / "artifacts"

    summary = train_models(
        tracking_uri=f"file:{tracking_dir}",
        experiment_name="artpulse_test",
        n_samples=1800,
        seed=7,
        test_size=0.2,
        output_dir=str(output_dir),
    )

    os.environ["MLFLOW_TRACKING_URI"] = f"file:{tracking_dir}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "artpulse_test"
    os.environ["MODEL_URI"] = summary["best"]["model_uri"]
    os.environ["MODEL_URI_FILE"] = str(Path(output_dir) / "latest_model_uri.txt")

    import src.serve as serve

    importlib.reload(serve)

    with TestClient(serve.app) as client:
        yield client

    for key in [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MODEL_URI",
        "MODEL_URI_FILE",
    ]:
        os.environ.pop(key, None)


def test_health_and_ready(api_client):
    health = api_client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] in {"ok", "degraded"}

    ready = api_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_metadata(api_client):
    resp = api_client.get("/metadata")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["labels"]
    assert payload["feature_order"]


def test_predict(api_client):
    payload = {
        "rows": [
            {
                "hue_mean": 0.62,
                "sat_mean": 0.55,
                "val_mean": 0.70,
                "contrast": 0.40,
                "edges": 0.12,
            }
        ]
    }
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["labels"]) == 1
    assert len(data["predictions"]) == 1


def test_predict_invalid_payload(api_client):
    payload = {
        "rows": [
            {
                "hue_mean": 4.0,
                "sat_mean": 0.55,
                "val_mean": 0.70,
                "contrast": 0.40,
                "edges": 0.12,
            }
        ]
    }
    resp = api_client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_image(api_client):
    image = Image.new("RGB", (96, 96), (240, 80, 90))
    buf = BytesIO()
    image.save(buf, format="PNG")

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    resp = api_client.post("/predict-image", json={"image_base64": encoded})
    assert resp.status_code == 200
    payload = resp.json()
    assert "label" in payload
    assert "features" in payload
    assert set(payload["features"].keys()) == {
        "hue_mean",
        "sat_mean",
        "val_mean",
        "contrast",
        "edges",
    }
