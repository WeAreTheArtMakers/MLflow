# Business Impact Report (One Page)

Use this report as client-facing proof of operational maturity.

## Output

- `artifacts/business_impact_onepager.md`

## Generate Locally

```bash
make business-impact-report \
  PUBLIC_API_URL="https://api.wearetheartmakers.com" \
  DEPLOY_LEAD_TIME_MIN="12" \
  ROLLBACK_TIME_MIN="2"
```

## Included KPIs

- Accuracy
- Macro F1
- P95 latency
- Error rate
- Drift score
- Deploy lead time
- Rollback time

## Data Sources

- `artifacts/training_summary.json`
- `artifacts/drift_report.json`
- `loadtest/k6/k6-summary.json` (optional)
- `artifacts/release_gate_report.json` (optional)

If load test or gate artifacts are missing, the generator writes `n/a` for unavailable fields.
