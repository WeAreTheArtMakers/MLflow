from fastapi.testclient import TestClient
from src.serve import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    payload = {"rows":[{"hue_mean":0.62,"sat_mean":0.55,"val_mean":0.70,"contrast":0.40,"edges":0.12}]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "labels" in data
    assert len(data["labels"]) == 1
