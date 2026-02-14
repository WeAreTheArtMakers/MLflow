#!/usr/bin/env bash
set -eo pipefail

REPO="${1:-WeAreTheArtMakers/MLflow}"
ENVS=(staging production)

REPO_REQUIRED_VARS=(MODEL_NAME)
ENV_REQUIRED_SECRETS=(MLFLOW_TRACKING_URI MLFLOW_REGISTRY_URI KUBE_CONFIG API_GATE_KEY)
ENV_REQUIRED_VARS=(AWS_ROLE_ARN AWS_REGION)
ENV_OPTIONAL_VARS=(S3_DATASET_URI S3_PREDICTION_LOG_URI ROLLOUT_STRATEGY_DEFAULT)

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 1
fi

if ! gh api rate_limit >/dev/null 2>&1; then
  echo "Cannot reach GitHub API right now. Try again." >&2
  exit 2
fi

fetch_lines() {
  local endpoint="$1"
  local jq_expr="$2"
  gh api "$endpoint" --jq "$jq_expr" 2>/dev/null || true
}

has_item() {
  local haystack="$1"
  local needle="$2"
  echo "$haystack" | tr ' ' '\n' | grep -Fxq "$needle"
}

echo "Repo: $REPO"

echo ""
echo "[Repo Variables]"
REPO_VARS="$(fetch_lines "repos/$REPO/actions/variables" '.variables[].name')"
for var in "${REPO_REQUIRED_VARS[@]}"; do
  if has_item "$REPO_VARS" "$var"; then
    echo "  OK   $var"
  else
    echo "  MISS $var"
  fi
done

for env in "${ENVS[@]}"; do
  echo ""
  echo "[Environment: $env]"
  ENV_SECRETS="$(fetch_lines "repos/$REPO/environments/$env/secrets" '.secrets[].name')"
  ENV_VARS="$(fetch_lines "repos/$REPO/environments/$env/variables" '.variables[].name')"

  echo "  Secrets:"
  for s in "${ENV_REQUIRED_SECRETS[@]}"; do
    if has_item "$ENV_SECRETS" "$s"; then
      echo "    OK   $s"
    else
      echo "    MISS $s"
    fi
  done

  echo "  Vars:"
  for v in "${ENV_REQUIRED_VARS[@]}"; do
    if has_item "$ENV_VARS" "$v"; then
      echo "    OK   $v"
    else
      echo "    MISS $v"
    fi
  done

  echo "  Env-specific required vars:"
  if [[ "$env" == "staging" ]]; then
    if has_item "$ENV_VARS" "STAGING_API_URL"; then
      echo "    OK   STAGING_API_URL"
    else
      echo "    MISS STAGING_API_URL"
    fi
    if has_item "$ENV_VARS" "STAGING_PROMETHEUS_URL"; then
      echo "    OK   STAGING_PROMETHEUS_URL"
    else
      echo "    MISS STAGING_PROMETHEUS_URL"
    fi
  else
    if has_item "$ENV_VARS" "PROD_API_URL"; then
      echo "    OK   PROD_API_URL"
    else
      echo "    MISS PROD_API_URL"
    fi
    if has_item "$ENV_VARS" "PROD_PROMETHEUS_URL"; then
      echo "    OK   PROD_PROMETHEUS_URL"
    else
      echo "    MISS PROD_PROMETHEUS_URL"
    fi
  fi

  echo "  Optional vars:"
  for ov in "${ENV_OPTIONAL_VARS[@]}"; do
    if has_item "$ENV_VARS" "$ov"; then
      echo "    OK   $ov"
    else
      echo "    INFO $ov (optional)"
    fi
  done
done

cat <<'EOF'

Recommended set commands:
  gh variable set MODEL_NAME -R WeAreTheArtMakers/MLflow -b "artpulse-classifier"

  gh secret set MLFLOW_TRACKING_URI -R WeAreTheArtMakers/MLflow -e staging
  gh secret set MLFLOW_REGISTRY_URI -R WeAreTheArtMakers/MLflow -e staging
  gh secret set KUBE_CONFIG -R WeAreTheArtMakers/MLflow -e staging
  gh secret set API_GATE_KEY -R WeAreTheArtMakers/MLflow -e staging
  gh variable set STAGING_API_URL -R WeAreTheArtMakers/MLflow -e staging -b "https://api.staging.wearetheartmakers.com"
  gh variable set STAGING_PROMETHEUS_URL -R WeAreTheArtMakers/MLflow -e staging -b "https://prometheus.staging.wearetheartmakers.com"
  gh variable set AWS_ROLE_ARN -R WeAreTheArtMakers/MLflow -e staging -b "arn:aws:iam::<account-id>:role/<role-name>"
  gh variable set AWS_REGION -R WeAreTheArtMakers/MLflow -e staging -b "eu-central-1"

  gh secret set MLFLOW_TRACKING_URI -R WeAreTheArtMakers/MLflow -e production
  gh secret set MLFLOW_REGISTRY_URI -R WeAreTheArtMakers/MLflow -e production
  gh secret set KUBE_CONFIG -R WeAreTheArtMakers/MLflow -e production
  gh secret set API_GATE_KEY -R WeAreTheArtMakers/MLflow -e production
  gh variable set PROD_API_URL -R WeAreTheArtMakers/MLflow -e production -b "https://api.wearetheartmakers.com"
  gh variable set PROD_PROMETHEUS_URL -R WeAreTheArtMakers/MLflow -e production -b "https://prometheus.wearetheartmakers.com"
  gh variable set AWS_ROLE_ARN -R WeAreTheArtMakers/MLflow -e production -b "arn:aws:iam::<account-id>:role/<role-name>"
  gh variable set AWS_REGION -R WeAreTheArtMakers/MLflow -e production -b "eu-central-1"
EOF
