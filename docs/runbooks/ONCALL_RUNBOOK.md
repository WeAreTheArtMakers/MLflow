# On-Call Runbook

## Rotation Model

- Primary on-call: 1 engineer.
- Secondary on-call: 1 backup engineer.
- Shift handover: daily with status notes.

## Handover Checklist

1. Open incidents and status.
2. Active canary/blue-green traffic percentages.
3. Latest drift score and retrain schedule status.
4. Pending production changes.

## Minimum Monitoring Set

- API health/readiness.
- Error rate (5xx, auth failures, rate-limit spikes).
- P95/P99 latency.
- Drift score trend.
- Retrain pipeline success/failure.

## Shift Workflow

1. Confirm alerting channels are healthy.
2. Review `/ops/summary` every shift start.
3. Run smoke request:

```bash
curl -sS http://<api>/health
curl -sS http://<api>/ready
```

4. Escalate per `ESCALATION_MATRIX.md` if thresholds are crossed.

## Operational Hygiene

- Record all manual interventions in incident log.
- Do not bypass change controls for non-SEV emergencies.
- Rotate temporary credentials immediately after incident closure.
