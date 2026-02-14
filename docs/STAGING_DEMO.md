# Live Demo (Staging)

Use this file for portfolio/client demo sessions.

## Staging URL

- API base URL: `http://localhost:8000` (local staging baseline)
- Management panel: `http://localhost:8000/admin`
- Health: `http://localhost:8000/health`
- Public demo status: `http://localhost:8000/demo/status`

For internet-facing demos, replace with your public endpoint:

- `https://api.staging.<your-domain>/`

## Demo Request/Response

Sample request:

- `examples/staging_predict_request.json`

Sample response:

- `examples/staging_predict_response.json`

Public demo request/response:

- `examples/public_demo_request.json`
- `examples/public_demo_response.json`

Run demo call:

```bash
make staging-demo API_URL="http://localhost:8000" ARTPULSE_API_KEY="replace-with-key"
```

Run public demo endpoint:

```bash
make demo-token API_URL="http://localhost:8000" ARTPULSE_API_KEY="replace-with-key" ARTPULSE_DEMO_SUBJECT="portfolio-client"
make demo-status API_URL="http://localhost:8000" ARTPULSE_API_KEY="replace-with-key" ARTPULSE_DEMO_SUBJECT="portfolio-client"
make demo-predict API_URL="http://localhost:8000" ARTPULSE_API_KEY="replace-with-key" ARTPULSE_DEMO_SUBJECT="portfolio-client"
```

## Demo Narrative (Client-Friendly)

1. Show `GET /ops/summary` to explain current model, drift score, and retrain timestamp.
2. Show controlled rollout mode (`single/canary/blue_green`) and traffic percentage.
3. Run `staging-demo` request and explain prediction + model URI served.
4. Run `/demo/token` then `/demo/predict` to demonstrate short-lived signed demo tokens + throttling.
5. Open Grafana dashboard and connect business KPIs to operational telemetry.
