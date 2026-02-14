# Escalation Matrix

## Contacts by Role

- `L1` On-call engineer: first response and containment.
- `L2` Senior MLOps engineer: deep diagnosis, model/deploy decisions.
- `L3` Platform/Security lead: infrastructure/security escalation.
- `Business owner`: client communication and stakeholder alignment.

## Escalation Policy

| Severity | Escalate To | Max Escalation Time | Communication Cadence |
| --- | --- | --- | --- |
| SEV-1 | L2 + L3 + Business owner | 10 minutes | Every 15 minutes |
| SEV-2 | L2 | 30 minutes | Every 30 minutes |
| SEV-3 | L1 backlog + daily sync | 1 business day | Daily |

## Trigger Conditions

- Error rate above threshold for 5+ minutes.
- P95 latency sustained above SLA threshold.
- Security/auth anomaly (credential leak, abusive traffic).
- Drift score above agreed threshold and quality degradation confirmed.

## Client Communication Template

1. Incident ID and severity.
2. Impact scope and affected services.
3. Immediate mitigation in progress.
4. Next update timestamp.
