# Secure GitHub Secrets/Vars Setup (No Security Compromise)

This document hardens GitHub Actions setup for `/Users/bg/MLflow/MLflow`.

## Security Principles Applied

- Least privilege workflow permissions.
- Environment-scoped secrets (`staging`, `production`) instead of broad repo secrets for deployment-critical values.
- Branch-policy protected environments.
- Input validation before deployment commands.
- AWS OIDC federation (no long-lived `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).

## 1) What is already done

- Environments created:
  - `staging`
  - `production`
- Branch protection policy enabled on both environments (`protected_branches=true`).
- Workflow hardening applied:
  - `/Users/bg/MLflow/MLflow/.github/workflows/deploy-promotion.yml`
  - `/Users/bg/MLflow/MLflow/.github/workflows/retrain.yml`
  - `/Users/bg/MLflow/MLflow/.github/workflows/build-image.yml`
  - `/Users/bg/MLflow/MLflow/.github/workflows/ci.yml`

## 2) Required Variables/Secrets

Repository Variable:

- `MODEL_NAME` (default: `artpulse-classifier`)

Environment secrets (`staging` and `production`):

- `MLFLOW_TRACKING_URI`
- `MLFLOW_REGISTRY_URI`

Environment secret (`production`):

- `KUBE_CONFIG` (base64 kubeconfig)

Environment vars (`staging` and `production`) for OIDC retraining:

- `AWS_ROLE_ARN`
- `AWS_REGION`

Optional environment vars:

- `S3_DATASET_URI`
- `S3_PREDICTION_LOG_URI`

## 3) Set values securely

Set repo var:

```bash
gh variable set MODEL_NAME -R WeAreTheArtMakers/MLflow -b "artpulse-classifier"
```

Set environment secrets (interactive):

```bash
/Users/bg/MLflow/MLflow/scripts/gh_set_secret_prompt.sh WeAreTheArtMakers/MLflow staging MLFLOW_TRACKING_URI
/Users/bg/MLflow/MLflow/scripts/gh_set_secret_prompt.sh WeAreTheArtMakers/MLflow staging MLFLOW_REGISTRY_URI

/Users/bg/MLflow/MLflow/scripts/gh_set_secret_prompt.sh WeAreTheArtMakers/MLflow production MLFLOW_TRACKING_URI
/Users/bg/MLflow/MLflow/scripts/gh_set_secret_prompt.sh WeAreTheArtMakers/MLflow production MLFLOW_REGISTRY_URI
/Users/bg/MLflow/MLflow/scripts/gh_set_secret_prompt.sh WeAreTheArtMakers/MLflow production KUBE_CONFIG
```

Set environment vars:

```bash
gh variable set AWS_ROLE_ARN -R WeAreTheArtMakers/MLflow -e staging -b "arn:aws:iam::<account-id>:role/<oidc-role>"
gh variable set AWS_REGION -R WeAreTheArtMakers/MLflow -e staging -b "eu-central-1"

gh variable set AWS_ROLE_ARN -R WeAreTheArtMakers/MLflow -e production -b "arn:aws:iam::<account-id>:role/<oidc-role>"
gh variable set AWS_REGION -R WeAreTheArtMakers/MLflow -e production -b "eu-central-1"
```

Optional S3 vars:

```bash
gh variable set S3_DATASET_URI -R WeAreTheArtMakers/MLflow -e staging -b "s3://<bucket>/artpulse/images"
gh variable set S3_PREDICTION_LOG_URI -R WeAreTheArtMakers/MLflow -e staging -b "s3://<bucket>/artpulse/prediction_events.jsonl"

gh variable set S3_DATASET_URI -R WeAreTheArtMakers/MLflow -e production -b "s3://<bucket>/artpulse/images"
gh variable set S3_PREDICTION_LOG_URI -R WeAreTheArtMakers/MLflow -e production -b "s3://<bucket>/artpulse/prediction_events.jsonl"
```

## 4) Verify setup

```bash
/Users/bg/MLflow/MLflow/scripts/gh_audit_config.sh WeAreTheArtMakers/MLflow
```

## 5) Go live sequence

1. Trigger `Build And Push Image` workflow.
2. Trigger `Drift Monitor And Retrain` on `staging` and validate challenger metrics.
3. Trigger `Promote Model And Deploy` on `staging` (`promote_alias=true`, `challenger -> champion`).
4. Run smoke tests against staging API.
5. Repeat `Promote Model And Deploy` for `production` with explicit `image_tag`.

## 6) Extra hardening recommended (manual)

- Add required reviewers on `production` environment.
- Disable admin bypass in production environment if policy allows.
- Enforce branch protection + required status checks (`CI`).
- Rotate MLflow/Kubernetes credentials periodically.
