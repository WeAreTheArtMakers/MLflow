# Uptime + Alert Setup (1-Minute Checks)

Use this runbook to detect outages before clients do.

## Targets

- Production health: `https://api.wearetheartmakers.com/health`
- Production readiness: `https://api.wearetheartmakers.com/ready`
- (Optional) Staging health: `https://api.staging.wearetheartmakers.com/health`

## Minimum Policy

- Check interval: `1 minute`
- Timeout: `10 seconds`
- Incident trigger: `2 consecutive failures`
- Recovery trigger: `1 success`
- Channels: email + Slack webhook

## Better Stack (example)

1. Create monitor `artpulse-prod-health` for `/health`.
2. Create monitor `artpulse-prod-ready` for `/ready`.
3. Add alert escalation:
   - Stage 1 (immediate): Email
   - Stage 2 (+5 min unresolved): Slack
4. Add maintenance window for planned deployments.

## UptimeRobot (example)

1. Add `HTTPS` monitor for `/health` with `1 minute` interval.
2. Add second monitor for `/ready`.
3. Configure alert contacts:
   - Primary: ops email
   - Secondary: Slack integration
4. Enable alert repetition every `5 minutes` while down.

## Alert Message Template

Use a short payload in alerts:

- Service: `artpulse-api`
- Environment: `production`
- URL: failing URL
- Last status code
- Timestamp (UTC)
- Suggested action: "Check /health, /ready, rollout status, rollback if needed"

## Validation (after setup)

1. Trigger one controlled failure in staging and confirm alert arrives.
2. Verify recovery notifications are sent.
3. Confirm on-call backup recipient also receives alerts.

## Inputs Needed From You (to enable now)

Provide these once and monitoring can be activated immediately:

1. Monitoring provider choice: `Better Stack` or `UptimeRobot`.
2. Primary alert email (for warnings).
3. Secondary/on-call backup email (for critical).
4. Slack incoming webhook URL (or channel + app integration method).
5. Escalation policy:
   - Warning trigger: default `2 failures`
   - Critical trigger: default `5 failures`
   - Repeat interval while down: default `5 minutes`
6. Maintenance window preference (UTC) for planned deploys.

Current endpoints to monitor:

- Production:
  - `https://api.wearetheartmakers.com/health`
  - `https://api.wearetheartmakers.com/ready`
- Staging:
  - `https://api.staging.wearetheartmakers.com/health`
  - `https://api.staging.wearetheartmakers.com/ready`
