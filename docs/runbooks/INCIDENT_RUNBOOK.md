# Incident Runbook (24/7)

## Severity Definition

- `SEV-1`: API outage, widespread failed inference, security incident.
- `SEV-2`: elevated error rate/latency, partial model quality degradation.
- `SEV-3`: non-critical defects, delayed batch/retrain operations.

## Initial Response Targets

- `SEV-1`: acknowledge in <= 5 minutes.
- `SEV-2`: acknowledge in <= 15 minutes.
- `SEV-3`: acknowledge in <= 4 hours.

## Triage Checklist

1. Confirm alert source (Prometheus/Grafana/API error logs).
2. Validate impact scope:
   - affected endpoint(s)
   - error rate
   - latency change
   - affected clients/regions
3. Classify severity and open incident channel.
4. Assign incident commander + scribe.

## Containment Actions

1. If drift or bad predictions are detected:
   - reduce canary/secondary traffic to `0%`
   - switch to stable alias (`champion`) only
2. If deployment regression:
   - rollback image tag to previous known-good release
3. If auth abuse or attack:
   - rotate API keys
   - tighten rate limits
   - block offending network ranges (gateway/WAF)

## Recovery Criteria

- Error rate returns below SLO threshold.
- P95 latency back in expected range.
- Canary checks pass for at least 30 minutes.
- Incident commander declares resolved.

## Postmortem (within 48h)

- Timeline with timestamps.
- Root cause and contributing factors.
- Corrective and preventive actions (CAPA).
- Ownership and due dates.
