# Staging -> Production Promotion and Rollback Test

Use this runbook to keep deployments controlled on the 2-service free-plan setup.

## Scope

- Staging API: `https://api.staging.wearetheartmakers.com`
- Production API: `https://api.wearetheartmakers.com`
- Workflow: `.github/workflows/deploy-promotion.yml`

## Pre-Flight Checklist

1. Staging smoke checks pass:
   - `GET /health`
   - `GET /ready`
   - `/demo/token -> /demo/status -> /demo/predict`
2. New image tag exists in GHCR.
3. `run_quality_gate=true` is planned for promotion.
4. `API_GATE_KEY`, `STAGING_API_URL`, `PROD_API_URL`, `STAGING_PROMETHEUS_URL`, `PROD_PROMETHEUS_URL` are set in GitHub Environment.

## Normal Promotion (Manual)

Trigger `Promote Model And Deploy` with:

- `environment=production`
- `image_tag=sha-<commit>`
- `promote_alias=true`
- `source_alias=challenger`
- `target_alias=champion`
- `rollout_strategy=single` (or `canary`)
- `run_quality_gate=true`
- `gate_p95_ms=180`
- `gate_error_rate_max=0.01`
- `gate_drift_score_max=2.5`
- `gate_min_duration_min=15`

Expected result:

- workflow finishes green
- deployment remains on new revision

## Controlled Gate-Fail Rollback Drill (Recommended Weekly)

Goal: prove rollback works before a real incident.

Trigger same workflow, but force a failure threshold:

- `environment=staging`
- `run_quality_gate=true`
- `gate_p95_ms=1`
- keep other thresholds unchanged

Expected result:

1. `Run release quality gate` fails.
2. `Auto rollback if quality gate failed` runs successfully.
3. Workflow ends failed with message: `Release quality gate failed; deployment rolled back.`
4. Staging `/health` returns `status=ok` after rollback.

## Evidence to Save

For each promotion and rollback drill, capture:

1. Workflow run URL
2. Gate report artifact/log lines
3. Deployed image tag before/after
4. Post-check screenshots:
   - `/health`
   - `/admin`

Store evidence in the proof package for client-facing trust.
