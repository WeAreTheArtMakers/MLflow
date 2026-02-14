# Live Demo Script (60 Seconds)

Use this script in client calls to demonstrate production readiness quickly.

## Prerequisites

- Base URL: `https://api.wearetheartmakers.com`
- Valid API key with permission to mint demo tokens
- `jq` installed locally

## 60-Second Flow

1. Show service health:

```bash
curl -sS https://api.wearetheartmakers.com/health | jq '{status, model_loaded, rollout, demo}'
```

2. Mint short-lived demo token:

```bash
API_KEY="<replace-with-api-key>"
TOKEN=$(curl -sS -X POST "https://api.wearetheartmakers.com/demo/token" \
  -H "content-type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"subject":"portfolio-demo","ttl_seconds":600}' | jq -r '.access_token')
```

3. Show demo status (token-protected endpoint):

```bash
curl -sS "https://api.wearetheartmakers.com/demo/status" \
  -H "Authorization: Bearer $TOKEN" | jq
```

4. Run one prediction:

```bash
curl -sS -X POST "https://api.wearetheartmakers.com/demo/predict" \
  -H "content-type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @examples/public_demo_request.json | jq
```

5. Open ops panel in browser:

- `https://api.wearetheartmakers.com/admin`

6. Optional no-CLI browser demo:

- Open `https://api.wearetheartmakers.com/`
- Use the `Try It (No CLI)` card
- Click `Mint Token` -> `Demo Status` -> `Demo Predict`

## Talking Points (Client-Friendly)

1. The model is live, monitored, and versioned.
2. Public demo traffic uses short-lived signed tokens.
3. Rollout/rollback logic is production-safe.
4. Operational telemetry is visible in the admin panel.
