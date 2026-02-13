#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <repo> <environment> <secret_name>" >&2
  echo "Example: $0 WeAreTheArtMakers/MLflow production KUBE_CONFIG" >&2
  exit 1
fi

REPO="$1"
ENV_NAME="$2"
SECRET_NAME="$3"

read -r -s -p "Enter value for $SECRET_NAME ($ENV_NAME): " SECRET_VALUE
echo

if [[ -z "$SECRET_VALUE" ]]; then
  echo "Empty value is not allowed" >&2
  exit 1
fi

printf '%s' "$SECRET_VALUE" | gh secret set "$SECRET_NAME" -R "$REPO" -e "$ENV_NAME" -b-
unset SECRET_VALUE

echo "Secret set: $SECRET_NAME in environment $ENV_NAME"
