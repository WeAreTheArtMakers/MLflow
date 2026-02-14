# Rollback Runbook

## When to Roll Back

- `SEV-1/SEV-2` incidents caused by latest model or deployment.
- Canary/blue-green validation fails.
- Drift score spikes and prediction quality drops.

## Rollback Paths

## 1) Model rollback (fastest)

1. Identify stable model version from MLflow Registry.
2. Re-point alias:

```bash
python3 -m src.model_registry \
  --tracking-uri "$MLFLOW_TRACKING_URI" \
  --registry-uri "$MLFLOW_REGISTRY_URI" \
  set-alias --model-name artpulse-classifier --version <stable_version> --alias champion
```

3. Reload API model:

```bash
curl -X POST http://<api>/reload-model -H "x-api-key: <key>"
```

## 2) Rollback canary/blue-green traffic

1. Set secondary traffic to `0%`.
2. Keep rollout mode active for diagnostics or switch to `single`.
3. Confirm via `/health` and `/ops/summary`.

## 3) Deployment/image rollback

```bash
kubectl -n artpulse rollout undo deployment/artpulse
kubectl -n artpulse rollout status deployment/artpulse --timeout=180s
```

## Verification

- `GET /health` returns `status=ok`
- `GET /ready` returns `ready`
- Error rate and latency dashboards normalize
- Client smoke tests pass
