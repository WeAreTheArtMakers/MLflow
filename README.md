# ArtPulse - Real Image MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

## About

ArtPulse is an end-to-end MLOps reference project that takes an AI/ML model from idea to production.

This repository demonstrates a full delivery story you can present to clients and hiring teams:

- Real image feature extraction and model training
- MLflow experiment tracking, metrics logging, and artifact management
- Model Registry alias flow (`challenger -> champion`) for controlled promotion
- Production-ready FastAPI inference endpoints (`/predict`, `/predict-image`)
- Docker and Kubernetes deployment
- Drift monitoring and scheduled retraining automation

## Turkish Documentation

If you prefer Turkish documentation, see:

- `docs/README_TR.md`

## Who This Is For

- Companies that need production ML, not just notebook experiments
- Teams building an MVP MLOps platform quickly
- Individual engineers building a strong technical portfolio for client work

## Value Proposition (Client-Facing)

"With ArtPulse, I deliver not only model accuracy but also deployment, versioning, promotion/rollback, and operational monitoring in a working production pipeline."

## What Problems This Solves

This project directly answers:

- Did you train models on real data?
- Can you promote models with registry aliases in production?
- Do you have CI/CD for build and deployment promotion?
- Do you monitor drift and support periodic retraining?

Yes - all are implemented here.

## Model Summary

The model predicts an art style label from 5 compact visual features:

- `hue_mean`
- `sat_mean`
- `val_mean`
- `contrast`
- `edges`

Labels:

- `minimal`
- `neo-pop`
- `surreal`
- `monochrome`
- `vibrant`

## Quick Start

```bash
cd /path/to/MLflow
make install
```

### Synthetic quick demo

```bash
make demo
```

### Real image pipeline demo

```bash
make demo-image
```

These commands:

- train multiple candidate models
- log metrics/artifacts to MLflow
- select and mark the best model
- generate `artifacts/training_summary.json` and `artifacts/example_predictions.json`

## Real Image Pipeline

### Dataset layout

Use `examples/image_dataset_layout.txt` as the expected folder structure.

Required class folders:

- `minimal`
- `neo-pop`
- `surreal`
- `monochrome`
- `vibrant`

### Generate local sample dataset

```bash
make generate-images
```

Default output:

- `data/images/<label>/*.png`

### Train with real images

```bash
make train-images
```

Manual command:

```bash
.venv/bin/python -m src.train \
  --dataset-type image \
  --dataset-dir data/images \
  --register-best \
  --model-name artpulse-classifier \
  --model-alias champion
```

## Remote MLflow + Model Registry Alias

Example environment template:

- `examples/remote_env.example`

Example usage:

```bash
export MLFLOW_TRACKING_URI="http://mlflow.example.com"
export MLFLOW_REGISTRY_URI="http://mlflow.example.com"
export MODEL_NAME="artpulse-classifier"
export MODEL_ALIAS="champion"
```

### Load model by alias in API

```bash
export USE_MODEL_REGISTRY_ALIAS=true
make serve
```

Model URI in this mode:

- `models:/artpulse-classifier@champion`

### Promote alias

```bash
make promote-alias
```

Promotes `challenger -> champion`.

## API Usage

Start service:

```bash
make serve
```

Endpoints:

- `GET /health`
- `GET /ready`
- `POST /reload-model`
- `GET /metadata`
- `POST /predict` (tabular)
- `POST /predict-image` (base64 image payload)

Requests:

```bash
make predict
make predict-image
```

Prediction event log:

- `artifacts/prediction_events.jsonl`

## Drift Monitoring + Retraining

Generate drift report:

```bash
make monitor-drift
```

Output:

- `artifacts/drift_report.json`

Run periodic retraining job:

```bash
make retrain
```

## CI/CD Workflows

In `.github/workflows/`:

- `ci.yml`: tests + synthetic/image training smoke checks
- `build-image.yml`: GHCR image build/push
- `deploy-promotion.yml`: model alias promotion + k8s rollout
- `retrain.yml`: scheduled/manual drift + retrain pipeline

For secure GitHub secrets/variables setup:

- `docs/GITHUB_SECURE_SETUP.md`

## Docker and Kubernetes

Docker:

```bash
make docker-build
make docker-run
```

Kubernetes:

```bash
make k8s-apply
kubectl -n artpulse port-forward svc/artpulse 8000:80
curl -sS http://localhost:8000/health
```

## Extended Project Scope (Added)

The following services are now part of the project offering roadmap:

- New data collection infrastructure setup
- Frontend productization (dashboard/UI layer)
- Long-term 24/7 operational NOC/SRE service model

## Portfolio Positioning

Recommended positioning for CV/portfolio/GitHub profile:

- Role: `ML Engineer / MLOps Engineer`
- Focus: `Model lifecycle ownership (train -> registry -> deploy -> monitor)`
- Deliverables: `API, containerization, Kubernetes rollout, drift reporting, retraining automation`

Suggested one-liners:

1. "Delivered a real-image classification pipeline with production-ready MLOps architecture."
2. "Implemented controlled model promotion using MLflow Model Registry aliases."
3. "Enabled operational continuity with drift monitoring and retraining automation."

## Client-Facing Development Recommendations

### Phase 1 - Immediate business value (1-2 weeks)

- Domain-specific data validation rules
- API auth and rate limiting (JWT + gateway)
- SLO/SLI dashboard (latency, error rate, model freshness)

### Phase 2 - Enterprise scaling (2-4 weeks)

- Canary/A-B model rollout
- Feature store integration
- Automated quality gates before promotion

### Phase 3 - Enterprise operations (4+ weeks)

- Audit trail and lineage reporting
- PII governance and retention policies
- On-prem / VPC deployment blueprint
- 24/7 NOC/SRE runbook and on-call model

## Important Files

```text
src/features.py               # synthetic + real image feature extraction
src/generate_image_dataset.py # sample real-image style dataset generator
src/train.py                  # train, compare, registry registration
src/serve.py                  # API, image prediction, event logging
src/monitor_drift.py          # drift report generation
src/retrain_job.py            # periodic retraining job
src/model_registry.py         # alias promotion utilities
```

## Validation

```bash
make test
```

Coverage includes:

- API health/readiness/predict/predict-image
- real image dataset extraction
- image-based training flow

---

Built with <3 WeAreTheArtMakers
