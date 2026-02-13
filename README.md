# ArtPulse - Creative MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

ArtPulse is an end-to-end MLOps portfolio project that shows how to move a model from:

`training -> experiment tracking -> model selection -> API serving -> containerization -> Kubernetes deployment`

## Why this project matters

If someone asks: "Have you deployed AI/ML models into production?"
this repo gives a concrete answer:

- Multiple candidate models are trained and compared.
- Metrics and artifacts are tracked in MLflow.
- The best run is automatically tagged as deployment-ready.
- A FastAPI service loads the best model and serves predictions.
- The service is packaged with Docker and deployed to Kubernetes with probes and autoscaling.

## What the model predicts

ArtPulse predicts an art style/mood class from compact visual features:

- `hue_mean`
- `sat_mean`
- `val_mean`
- `contrast`
- `edges`

Output labels:

- `minimal`
- `neo-pop`
- `surreal`
- `monochrome`
- `vibrant`

V1 uses synthetic data for reproducible demos. You can later replace the synthetic feature generator with real image feature extraction.

## Architecture

```text
Training (src.train)
  |- train 3 candidate models
  |- log runs/metrics/models to MLflow
  |- choose best by f1_macro
  |- tag best run: deployment_ready=true
  v
MLflow Tracking (mlruns)
  |- experiment history
  |- model artifacts
  v
Serving (src.serve / FastAPI)
  |- discover best model URI
  |- load model via MLflow pyfunc
  |- expose /health /ready /metadata /predict
  v
Docker
  |- build image
  |- bootstrap model during build
  v
Kubernetes
  |- Deployment + Service + HPA
```

## Repository structure

```text
src/
  features.py     # synthetic dataset + shared feature schema
  train.py        # train/compare/log candidate models
  serve.py        # FastAPI inference service
k8s/
  namespace.yaml
  deployment.yaml
  service.yaml
  hpa.yaml
tests/
  test_api.py
Dockerfile
Makefile
requirements.txt
```

## Local setup

```bash
make install
```

## Train and track models

```bash
make train
```

This will:

- train `logistic_regression`, `random_forest`, `hist_gradient_boosting`
- log all runs to MLflow
- select best run by `f1_macro` (then `accuracy`)
- mark best run with MLflow tag `deployment_ready=true`
- write:
  - `artifacts/latest_model_uri.txt`
  - `artifacts/training_summary.json`

Open MLflow UI:

```bash
make ui
# http://localhost:5000
```

## Run API locally

```bash
make serve
# http://localhost:8000
```

Endpoints:

- `GET /health` - liveness and model load state
- `GET /ready` - readiness (503 if model cannot be loaded)
- `GET /metadata` - labels and feature order
- `POST /predict` - prediction endpoint

Example prediction:

```bash
make predict
```

Direct request example:

```bash
curl -sS -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "hue_mean": 0.62,
        "sat_mean": 0.55,
        "val_mean": 0.70,
        "contrast": 0.40,
        "edges": 0.12
      }
    ]
  }'
```

## Run tests

```bash
make test
```

Tests cover training + API health/readiness + valid/invalid prediction payloads.

## Docker

Build image:

```bash
make docker-build
```

Run container:

```bash
make docker-run
```

Notes:

- The Docker build runs training once and stores a ready model inside the image.
- API starts ready without external MLflow infra for demo purposes.

## Kubernetes

Apply manifests:

```bash
make k8s-apply
```

Access locally:

```bash
kubectl -n artpulse port-forward svc/artpulse 8000:80
curl -sS http://localhost:8000/health
```

HPA is configured in `k8s/hpa.yaml` based on CPU utilization.

## Production extension ideas

- Replace synthetic data with real image feature extraction pipeline.
- Move MLflow to remote backend (S3 + DB) and use Model Registry aliases.
- Add CI/CD for image build and deployment promotion.
- Add drift monitoring and periodic retraining automation.
