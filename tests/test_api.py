import base64
import importlib
import os
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.train import train_models

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"x-api-key": TEST_API_KEY}


def _set_env(values: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in values.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        os.environ[key] = value


@pytest.fixture(scope="module")
def trained_model_config(tmp_path_factory):
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

    return {
        "MLFLOW_TRACKING_URI": f"file:{tracking_dir}",
        "MLFLOW_EXPERIMENT_NAME": "artpulse_test",
        "MODEL_URI": summary["best"]["model_uri"],
        "MODEL_URI_FILE": str(Path(output_dir) / "latest_model_uri.txt"),
    }


@pytest.fixture(scope="module")
def api_client(trained_model_config):
    env_values = {
        **trained_model_config,
        "AUTH_REQUIRED": "true",
        "API_KEYS": TEST_API_KEY,
        "JWT_SECRET": "",
        "RATE_LIMIT_ENABLED": "true",
        "RATE_LIMIT_REQUESTS": "500",
        "RATE_LIMIT_WINDOW_SECONDS": "60",
    }
    previous = _set_env(env_values)

    import src.serve as serve

    importlib.reload(serve)

    with TestClient(serve.app) as client:
        yield client

    _restore_env(previous)


@pytest.fixture()
def rate_limited_client(trained_model_config):
    env_values = {
        **trained_model_config,
        "AUTH_REQUIRED": "true",
        "API_KEYS": TEST_API_KEY,
        "JWT_SECRET": "",
        "RATE_LIMIT_ENABLED": "true",
        "RATE_LIMIT_REQUESTS": "2",
        "RATE_LIMIT_WINDOW_SECONDS": "60",
    }
    previous = _set_env(env_values)

    import src.serve as serve

    importlib.reload(serve)

    with TestClient(serve.app) as client:
        yield client

    _restore_env(previous)


@pytest.fixture()
def canary_client(trained_model_config):
    env_values = {
        **trained_model_config,
        "AUTH_REQUIRED": "true",
        "API_KEYS": TEST_API_KEY,
        "JWT_SECRET": "",
        "RATE_LIMIT_ENABLED": "true",
        "RATE_LIMIT_REQUESTS": "200",
        "RATE_LIMIT_WINDOW_SECONDS": "60",
        "ROLLOUT_MODE": "canary",
        "CANARY_TRAFFIC_PERCENT": "100",
        "CANDIDATE_MODEL_URI": trained_model_config["MODEL_URI"],
    }
    previous = _set_env(env_values)

    import src.serve as serve

    importlib.reload(serve)

    with TestClient(serve.app) as client:
        yield client

    _restore_env(previous)


@pytest.fixture()
def demo_client(trained_model_config):
    env_values = {
        **trained_model_config,
        "AUTH_REQUIRED": "true",
        "API_KEYS": TEST_API_KEY,
        "DEMO_ENABLED": "true",
        "DEMO_JWK_CURRENT_KID": "demo-v1",
        "DEMO_JWK_KEYS_JSON": '{"demo-v1":"test-demo-secret"}',
        "DEMO_RATE_LIMIT_REQUESTS": "100",
        "DEMO_RATE_LIMIT_WINDOW_SECONDS": "60",
        "JWT_SECRET": "",
        "RATE_LIMIT_ENABLED": "true",
        "RATE_LIMIT_REQUESTS": "200",
        "RATE_LIMIT_WINDOW_SECONDS": "60",
    }
    previous = _set_env(env_values)

    import src.serve as serve

    importlib.reload(serve)

    with TestClient(serve.app) as client:
        yield client

    _restore_env(previous)


def test_health_and_ready(api_client):
    health = api_client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] in {"ok", "degraded"}

    ready = api_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_metadata(api_client):
    resp = api_client.get("/metadata", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["labels"]
    assert payload["feature_order"]


def test_metadata_requires_auth(api_client):
    resp = api_client.get("/metadata")
    assert resp.status_code == 401


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
    resp = api_client.post("/predict", json=payload, headers=AUTH_HEADERS)
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
    resp = api_client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_predict_image(api_client):
    image = Image.new("RGB", (96, 96), (240, 80, 90))
    buf = BytesIO()
    image.save(buf, format="PNG")

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    resp = api_client.post(
        "/predict-image",
        json={"image_base64": encoded},
        headers=AUTH_HEADERS,
    )
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


def test_metrics(api_client):
    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "artpulse_http_requests_total" in body
    assert "artpulse_predictions_total" in body
    assert "artpulse_model_loaded" in body
    assert "artpulse_rollout_routed_total" in body


def test_ops_summary_requires_auth(api_client):
    resp = api_client.get("/ops/summary")
    assert resp.status_code == 401


def test_ops_summary(api_client):
    resp = api_client.get("/ops/summary", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    assert "model" in payload
    assert "drift" in payload
    assert "rollout" in payload


def test_admin_panel(api_client):
    resp = api_client.get("/admin")
    assert resp.status_code == 200
    assert "ArtPulse Ops Console" in resp.text


def test_demo_endpoint_disabled_by_default(api_client):
    resp = api_client.get("/demo/status")
    assert resp.status_code == 404


def test_demo_status_requires_demo_token(demo_client):
    resp = demo_client.get("/demo/status")
    assert resp.status_code == 401


def test_demo_status_and_predict(demo_client):
    token_resp = demo_client.post(
        "/demo/token",
        json={"subject": "qa-demo", "ttl_seconds": 600},
        headers=AUTH_HEADERS,
    )
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]
    demo_headers = {"Authorization": f"Bearer {token}"}

    status = demo_client.get("/demo/status", headers=demo_headers)
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "ready"

    predict = demo_client.post(
        "/demo/predict",
        json={
            "row": {
                "hue_mean": 0.52,
                "sat_mean": 0.63,
                "val_mean": 0.71,
                "contrast": 0.34,
                "edges": 0.19,
            }
        },
        headers=demo_headers,
    )
    assert predict.status_code == 200
    result = predict.json()
    assert result["rollout_mode"] in {"single", "canary", "blue_green"}
    assert result["rollout_route"] in {"primary", "secondary"}
    assert len(result["demo_request_id"]) == 16


def test_rate_limit(rate_limited_client):
    payload = {
        "rows": [
            {
                "hue_mean": 0.35,
                "sat_mean": 0.22,
                "val_mean": 0.80,
                "contrast": 0.31,
                "edges": 0.15,
            }
        ]
    }

    resp1 = rate_limited_client.post("/predict", json=payload, headers=AUTH_HEADERS)
    resp2 = rate_limited_client.post("/predict", json=payload, headers=AUTH_HEADERS)
    resp3 = rate_limited_client.post("/predict", json=payload, headers=AUTH_HEADERS)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 429
    assert "retry-after" in {k.lower() for k in resp3.headers}


def test_canary_rollout_uses_secondary(canary_client):
    payload = {
        "rows": [
            {
                "hue_mean": 0.61,
                "sat_mean": 0.72,
                "val_mean": 0.66,
                "contrast": 0.39,
                "edges": 0.21,
            }
        ]
    }

    resp = canary_client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["rollout_mode"] == "canary"
    assert body["rollout_route"] == "secondary"
