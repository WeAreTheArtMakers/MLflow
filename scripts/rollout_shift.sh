#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <namespace> <deployment> <mode:single|canary|blue_green> <secondary_percent> <active_color>"
  exit 1
fi

NAMESPACE="$1"
DEPLOYMENT="$2"
MODE="$3"
PERCENT="$4"
ACTIVE_COLOR="$5"

if [[ ! "$MODE" =~ ^(single|canary|blue_green)$ ]]; then
  echo "Invalid mode: $MODE"
  exit 2
fi

if [[ ! "$ACTIVE_COLOR" =~ ^(blue|green)$ ]]; then
  echo "Invalid active_color: $ACTIVE_COLOR"
  exit 2
fi

if [[ ! "$PERCENT" =~ ^([0-9]{1,3})(\.[0-9]+)?$ ]]; then
  echo "Invalid secondary_percent: $PERCENT"
  exit 2
fi

awk "BEGIN { exit !($PERCENT >= 0 && $PERCENT <= 100) }" || {
  echo "secondary_percent must be between 0 and 100"
  exit 2
}

kubectl -n "$NAMESPACE" set env deployment/"$DEPLOYMENT" \
  ROLLOUT_MODE="$MODE" \
  CANARY_TRAFFIC_PERCENT="$PERCENT" \
  BLUE_GREEN_TRAFFIC_PERCENT="$PERCENT" \
  ACTIVE_COLOR="$ACTIVE_COLOR"

kubectl -n "$NAMESPACE" rollout status deployment/"$DEPLOYMENT" --timeout=180s
echo "Rollout updated: mode=$MODE secondary_percent=$PERCENT active_color=$ACTIVE_COLOR"
