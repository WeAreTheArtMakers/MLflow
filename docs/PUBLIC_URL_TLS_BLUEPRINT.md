# Public URL + TLS Blueprint (Cloudflare + NGINX + cert-manager)

This guide matches the current ArtPulse manifests for:

- `api.staging.wearetheartmakers.com`
- `api.wearetheartmakers.com`

## Target Architecture

1. `Cloudflare DNS` exposes staging and production API hosts.
2. `ingress-nginx` terminates TLS and routes to `artpulse` service.
3. `cert-manager` issues certs with Let's Encrypt (`dns01`, Cloudflare API token).
4. `oauth2-proxy` protects `/admin` and `/ops/summary` behind OIDC.
5. Separate `/demo` ingress applies stricter WAF and CIDR allowlist.

## Manifest Set

- TLS issuers: `k8s/cert-manager-clusterissuer.yaml`
- Certificates:
  - `k8s/certificate-staging.yaml`
  - `k8s/certificate-production.yaml`
- Primary ingress:
  - `k8s/ingress-staging.yaml`
  - `k8s/ingress-production.yaml`
- Demo ingress (strict):
  - `k8s/ingress-demo-staging.yaml`
  - `k8s/ingress-demo-production.yaml`
- OIDC ingress:
  - `k8s/ingress-oauth2-staging.yaml`
  - `k8s/ingress-oauth2-production.yaml`
- OAuth2 proxy:
  - `k8s/oauth2-proxy-service.yaml`
  - `k8s/oauth2-proxy-deployment.yaml`
  - `k8s/oauth2-proxy-secrets.example.yaml`
- Cloudflare token examples:
  - `k8s/cert-manager-cloudflare-secrets.example.yaml`
- Network policy:
  - `k8s/network-policy.yaml`

## Deployment Sequence

1. Install `ingress-nginx` and `cert-manager`.
2. Create Cloudflare API token secrets in `cert-manager` namespace (staging + production).
3. Apply issuers and certificates:

```bash
kubectl apply -f k8s/cert-manager-clusterissuer.yaml
kubectl apply -f k8s/certificate-staging.yaml
kubectl apply -f k8s/certificate-production.yaml
```

4. Apply core app resources:

```bash
make k8s-apply
```

5. Apply public resources:

```bash
make k8s-apply-public-staging
make k8s-apply-public-production
```

## Secret Scope Recommendation (Staging vs Production)

Use separate secret objects per environment scope (and ideally separate namespaces/clusters):

- app/API secret (JWK + API keys)
- OIDC secret (`artpulse-oidc-secrets`)
- Cloudflare DNS solver secret (`cert-manager` namespace)

`k8s/secret.example.yaml` shows required app keys (`api_keys`, `demo_jwk_current_kid`, `demo_jwk_keys_json`, `jwt_secret`).

## Demo Endpoint Contract (Token-Based)

1. Mint short-lived token:
   - `POST /demo/token` with `x-api-key`
2. Call public endpoints with bearer token:
   - `GET /demo/status`
   - `POST /demo/predict`

Security controls:

- demo token TTL (default 10 minutes)
- JWK key rotation support (`DEMO_JWK_KEYS_JSON` + active `kid`)
- independent demo rate limits
- dedicated demo ingress WAF + CIDR allowlist

## Smoke Test (Public URL + TLS)

```bash
make check-public-tls \
  PUBLIC_API_URL="https://api.staging.wearetheartmakers.com" \
  ARTPULSE_API_KEY="<staging-api-key>" \
  ARTPULSE_DEMO_SUBJECT="portfolio-client"
```

This validates:

- HTTPS health endpoint
- certificate issuer/validity
- metrics exposure
- demo token mint + `/demo/status`

## Security Checklist

- Use separate staging/prod `ClusterIssuer` and certificate secrets.
- Keep Cloudflare token scope restricted to required zones only.
- Replace demo allowlist CIDRs with office/VPN egress IPs.
- Protect `/admin` with OIDC and forward access logs to SIEM.
- Enforce deploy quality gate + rollback in `deploy-promotion.yml`.
