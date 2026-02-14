#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <https-url> [api-key] [demo-subject]"
  exit 1
fi

URL="$1"
API_KEY="${2:-}"
DEMO_SUBJECT="${3:-portfolio-client}"

echo "[1/4] Health check"
curl -fsS "$URL/health" | python3 -m json.tool

echo "[2/4] TLS certificate check"
HOST="$(echo "$URL" | sed -E 's#https?://([^/]+).*#\1#')"
echo | openssl s_client -servername "$HOST" -connect "$HOST:443" 2>/dev/null | openssl x509 -noout -subject -issuer -dates

echo "[3/4] Metrics reachability"
curl -fsS "$URL/metrics" | sed -n '1,8p'

echo "[4/4] Optional demo endpoint"
if [[ -n "$API_KEY" ]]; then
  DEMO_TOKEN=$(curl -fsS -X POST "$URL/demo/token" \
    -H "content-type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d "{\"subject\":\"$DEMO_SUBJECT\",\"ttl_seconds\":600}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
  curl -fsS "$URL/demo/status" -H "Authorization: Bearer $DEMO_TOKEN" | python3 -m json.tool
else
  echo "Skipped (no api key provided to mint demo token)"
fi
