# Load Testing (k6)

This folder contains an API load test scenario for `POST /predict`.

## Quick run

```bash
make loadtest K6_API_URL="http://host.docker.internal:8000" ARTPULSE_API_KEY="replace-with-key"
make loadtest-report
```

Generated artifacts:

- `loadtest/k6/k6-summary.json`
- `artifacts/loadtest_report.md`

## Scenario profile

- Ramp to 10 users in 30s
- Ramp to 40 users in 1m
- Cool down to 0 users in 30s

Thresholds:

- `http_req_failed < 2%`
- `http_req_duration p95 < 200ms`
- `http_req_duration p99 < 350ms`
