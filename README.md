# ArtPulse - Production MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

## About

ArtPulse is an end-to-end MLOps portfolio project that demonstrates how to move AI/ML models from training to real operations.

Core capabilities:

- Real image feature extraction + model training
- MLflow experiment tracking and Model Registry alias management
- Secure FastAPI inference API (API key/JWT + rate-limit)
- Controlled model rollout (`single`, `canary`, `blue_green`)
- Prometheus-compatible metrics + Grafana dashboards
- Drift monitoring + retraining automation
- OIDC-protected admin surface (`/admin`) + audit logging hooks
- Docker + Kubernetes deployment
- Frontend operations panel (`/admin`) for live status

## Turkish Documentation

- `docs/README_TR.md`

## Live Endpoints (Production)

- Production API: `https://api.wearetheartmakers.com`
- Swagger: `https://api.wearetheartmakers.com/docs`
- Ops Panel: `https://api.wearetheartmakers.com/admin`

## Business Impact

Representative KPI targets for client-facing demos:

| KPI | Manual Baseline | ArtPulse Pipeline | Measurement Source |
| --- | --- | --- | --- |
| P95 inference latency | 180-250 ms | 45-90 ms | `/metrics` + Grafana latency panel |
| Accuracy | 0.86-0.90 | 0.93-0.95 | MLflow `metrics.accuracy` |
| Macro F1 | 0.83-0.88 | 0.91-0.93 | MLflow `metrics.f1_macro` |
| Deploy lead time | 45-90 min | 8-15 min | CI/CD workflow duration |
| Rollback time | 15-30 min | 1-3 min | Alias rollback + rollout restart |

These are benchmark targets for this reference architecture. Final numbers depend on infra scale and workload profile.

## Quick Start

```bash
make install
make demo-image
export ARTPULSE_API_KEY="replace-with-strong-key"
make serve ARTPULSE_API_KEY="$ARTPULSE_API_KEY"
```

## Real Image Training + Data Quality Gate

Train with image dataset and enforced pre-training checks:

```bash
make train-images DATASET_DIR=data/images
```

Quality controls (Great Expectations-style gate):

- feature schema validation
- finite value check (`NaN/inf` rejection)
- normalized feature range check (`0..1`)
- minimum sample + minimum per-label checks
- class imbalance warning

Artifacts:

- `artifacts/training_summary.json`
- `artifacts/data_quality_report.json`
- `artifacts/baseline_feature_stats.json`

To bypass quality checks (not recommended in production):

```bash
.venv/bin/python -m src.train --skip-quality-checks
```

## Controlled Rollout (Canary / Blue-Green)

Runtime traffic routing is controlled by env vars.

- `ROLLOUT_MODE=single|canary|blue_green`
- `CANARY_TRAFFIC_PERCENT` (0-100)
- `CANDIDATE_MODEL_ALIAS` (default `challenger`)
- `ACTIVE_COLOR=blue|green`
- `BLUE_GREEN_TRAFFIC_PERCENT` (0-100)

Example canary run (10% traffic to challenger):

```bash
make serve \
  ARTPULSE_API_KEY="$ARTPULSE_API_KEY" \
  ROLLOUT_MODE=canary \
  CANARY_TRAFFIC_PERCENT=10 \
  CANDIDATE_MODEL_ALIAS=challenger
```

Kubernetes rollout shift helper:

```bash
./scripts/rollout_shift.sh artpulse artpulse canary 25 blue
```

## API Security + Endpoints

Start API:

```bash
make serve ARTPULSE_API_KEY="$ARTPULSE_API_KEY"
```

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /admin` (frontend ops panel)
- `POST /demo/token` (auth, mints short-lived demo token)
- `GET /demo/status` (public demo, bearer token)
- `POST /demo/predict` (public demo, bearer token)
- `POST /reload-model` (auth)
- `GET /metadata` (auth)
- `GET /ops/summary` (auth)
- `POST /predict` (auth)
- `POST /predict-image` (auth)

API key request example:

```bash
curl -sS -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${ARTPULSE_API_KEY}" \
  -d '{"rows":[{"hue_mean":0.62,"sat_mean":0.55,"val_mean":0.70,"contrast":0.40,"edges":0.12}]}'
```

JWT mode is also supported (HS256) by setting `JWT_SECRET`.

Public demo mode is controlled separately:

- `DEMO_ENABLED=true|false`
- `DEMO_JWK_CURRENT_KID=<active-key-id>`
- `DEMO_JWK_KEYS_JSON=<json-map-of-kid-to-secret>`
- `DEMO_TOKEN_ISSUER=<token-issuer>`
- `DEMO_TOKEN_AUDIENCE=<token-audience>`
- `DEMO_TOKEN_TTL_SECONDS=<seconds>`
- `DEMO_RATE_LIMIT_REQUESTS`
- `DEMO_RATE_LIMIT_WINDOW_SECONDS`

## Frontend Operations Panel

Ops dashboard URL:

- `http://localhost:8000/admin`

Panel shows:

- primary/secondary model URIs
- alias versions (champion/challenger + blue/green when enabled)
- rollout mode and traffic percentage
- drift score and event count
- last retrain timestamp

## Observability (Prometheus + Grafana)

Start stack:

```bash
make monitoring-up
```

Services:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001` (`admin/admin`)
- API metrics: `http://localhost:8000/metrics`

Dashboard file:

- `observability/grafana/dashboards/artpulse-api-dashboard.json`

Stop stack:

```bash
make monitoring-down
```

## Load Testing (k6)

Run throughput/latency benchmark:

```bash
make loadtest K6_API_URL="http://host.docker.internal:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY"
make loadtest-report
```

Outputs:

- `loadtest/k6/k6-summary.json`
- `artifacts/loadtest_report.md`

## Drift Monitoring + Retraining

```bash
make monitor-drift
make retrain
```

Outputs:

- `artifacts/drift_report.json`
- new MLflow runs + optional challenger registration

## CI/CD and Promotion

Workflows in `.github/workflows/`:

- `ci.yml`
- `build-image.yml`
- `retrain.yml`
- `deploy-promotion.yml`

`deploy-promotion.yml` now supports rollout inputs:

- `rollout_strategy` (`single`, `canary`, `blue_green`)
- `rollout_secondary_percent`
- `candidate_alias`
- `active_color`
- `run_quality_gate` + gate thresholds (`gate_p95_ms`, `gate_error_rate_max`, `gate_drift_score_max`)

When the gate fails, workflow automatically executes `kubectl rollout undo` and marks deployment as failed.

Security setup guide:

- `docs/GITHUB_SECURE_SETUP.md`

## Docker + Kubernetes

Docker:

```bash
make docker-build
make docker-run
```

Kubernetes:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.example.yaml # edit values first
make k8s-apply
make k8s-apply-public-staging
make k8s-apply-public-production
```

Public URL + TLS blueprint:

- `docs/PUBLIC_URL_TLS_BLUEPRINT.md`
- `k8s/cert-manager-clusterissuer.yaml`
- `k8s/ingress-staging.yaml`
- `k8s/ingress-production.yaml`
- `k8s/ingress-demo-staging.yaml`
- `k8s/ingress-demo-production.yaml`
- `k8s/ingress-oauth2-staging.yaml`
- `k8s/ingress-oauth2-production.yaml`
- `k8s/network-policy.yaml`

## Live Demo (Staging)

Local staging baseline:

- API: `http://localhost:8000`
- Ops panel: `http://localhost:8000/admin`

Demo assets:

- `docs/STAGING_DEMO.md`
- `docs/LIVE_DEMO_SCRIPT.md`
- `examples/staging_predict_request.json`
- `examples/staging_predict_response.json`
- `examples/public_demo_request.json`
- `examples/public_demo_response.json`

Run staging demo request:

```bash
make staging-demo API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY"
```

Run public demo endpoint locally:

```bash
make serve ARTPULSE_API_KEY="$ARTPULSE_API_KEY" DEMO_ENABLED=true ARTPULSE_DEMO_KEY="replace-signing-secret"
make demo-token API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
make demo-status API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
make demo-predict API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
```

TLS/public smoke check helper:

```bash
make check-public-tls PUBLIC_API_URL="https://api.staging.<your-domain>" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
```

## Staging -> Production Promotion Policy

Use separate hostnames and a manual promotion gate:

- Staging API: `https://api.staging.wearetheartmakers.com`
- Production API: `https://api.wearetheartmakers.com`

Recommended rollout flow:

1. Deploy latest build to staging.
2. Run smoke tests on staging (`/health`, `/ready`, `/demo/token`, `/demo/predict`).
3. Trigger `.github/workflows/deploy-promotion.yml` with `environment=production` and `run_quality_gate=true`.
4. Promote only when quality gate passes.
5. Keep rollback path ready (`kubectl rollout undo` is automated in workflow on gate failure).

## Uptime Monitoring (1-Minute Ping + Email/Slack)

Set an external uptime monitor for production:

1. Create a monitor with URL `https://api.wearetheartmakers.com/health`.
2. Set check interval to `1 minute`.
3. Mark incident threshold as at least `2 consecutive failures`.
4. Enable alert channels:
   - Email (ops inbox)
   - Slack webhook/channel
5. Add a second monitor for `https://api.wearetheartmakers.com/ready` (optional but recommended).

Any provider works (Better Stack, UptimeRobot, Freshping, Pingdom). Keep checks HTTPS-only.

## 24/7 Operations Runbook Set

- `docs/runbooks/INCIDENT_RUNBOOK.md`
- `docs/runbooks/ROLLBACK_RUNBOOK.md`
- `docs/runbooks/ONCALL_RUNBOOK.md`
- `docs/runbooks/ESCALATION_MATRIX.md`

## Commercial Packaging (Starter / Growth / Enterprise)

| Package | Scope | SLA |
| --- | --- | --- |
| Starter | Single model pipeline, secure API, baseline monitoring, monthly support | 99.0% uptime, business-hours support |
| Growth | Staging+prod, canary/blue-green, drift automation, governance | 99.5% uptime, 8x5 support, P1 <= 1h |
| Enterprise | 24/7 NOC/SRE, compliance controls, incident drills, custom ops | 99.9% uptime, 24/7 support, P1 <= 15m |

Detailed package document:

- `docs/COMMERCIAL_PACKAGES.md`

## Development Recommendations

1. Replace handcrafted image features with embedding-based features (e.g. CLIP/ViT) and compare ROI.
2. Add online feature store integration to keep train/serve feature parity auditable.
3. Add shadow deployment mode for zero-risk candidate evaluation before canary traffic.
4. Add policy-based promotion gates (quality + latency + drift) before alias updates.
5. Add customer-level usage analytics in the ops panel for business reporting.

## Important Files

```text
src/train.py                   # training + MLflow tracking + quality gate
src/data_quality.py            # schema/range/distribution quality checks
src/serve.py                   # API auth, rollout routing, metrics, admin panel
src/monitor_drift.py           # drift score computation
src/retrain_job.py             # retraining automation
src/loadtest_report.py         # k6 summary -> markdown report
loadtest/k6/predict.js         # API performance test scenario
observability/docker-compose.yml
k8s/deployment.yaml
k8s/ingress-staging.yaml
k8s/ingress-production.yaml
k8s/ingress-demo-staging.yaml
k8s/ingress-demo-production.yaml
k8s/ingress-oauth2-staging.yaml
k8s/ingress-oauth2-production.yaml
k8s/cert-manager-clusterissuer.yaml
scripts/rollout_shift.sh
```

## Validation

```bash
make test
```

---

Built with <3 WeAreTheArtMakers
