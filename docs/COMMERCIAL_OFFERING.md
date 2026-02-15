# ArtPulse Commercial Offering

## What You Get

ArtPulse is delivered as a production ML operations service:

- model training + validation pipeline
- secure inference API with auth/rate-limit
- staged deployment (`staging -> production`)
- controlled rollout (`single`, `canary`, `blue_green`)
- monitoring, alerting, drift checks, retraining workflow
- ops visibility (`/admin`) and audit-ready runbooks

## Service Packages

| Package | Scope | SLA | Support Window |
| --- | --- | --- | --- |
| Starter | Single model, one production API, baseline monitoring | 99.0% | Business hours |
| Growth | Staging + production, canary/blue-green rollout, automated quality gates | 99.5% | 8x5 |
| Enterprise | Multi-team platform, compliance controls, managed incident response | 99.9% | 24/7 |

## Scope Matrix

| Capability | Starter | Growth | Enterprise |
| --- | --- | --- | --- |
| MLflow tracking and experiment governance | Yes | Yes | Yes |
| Model Registry aliases (champion/challenger) | Optional | Yes | Yes |
| OIDC/SSO for ops panel | Optional | Yes | Yes |
| Uptime alerting (email/slack) | Basic | Advanced | Advanced + escalation |
| Drift detection + retraining cadence | Monthly | Weekly | Weekly/daily (custom) |
| Runbooks (incident/rollback/on-call) | Basic | Standard | Full managed |

## Delivery Approach

1. Discovery and KPI definition (latency, accuracy, deploy lead time, rollback time)
2. Integration with client data sources and API contracts
3. Staging rollout and acceptance tests
4. Production go-live with guardrails
5. Hypercare + operational handover

## Commercial Terms

- Pricing is shared via proposal after scope discovery.
- SLA and response targets are finalized in contract.
- Security controls are adapted to client environment and compliance level.

## Contact

- Email: `ops@wearetheartmakers.com`
- Production API reference: `https://api.wearetheartmakers.com/docs`
- Ops demo: `https://api.wearetheartmakers.com/admin`
