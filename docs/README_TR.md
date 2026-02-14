# ArtPulse - Turkce Ozet Dokumani

Bu dosya, projenin Turkce ozetini verir.
Detayli teknik dokumantasyonun ana kaynagi: `README.md`.

## Proje Ozeti

ArtPulse; model egitimi, model kaydi, guvenli API servisleme, rollout, izleme ve operasyon sureclerini tek repoda gosteren uctan uca MLOps demosudur.

Yeni eklenen ana basliklar:

- Canary ve blue/green rollout (kontrollu trafik gecisi)
- Egitim oncesi veri kalite kapisi (GE-benzeri kontroller)
- k6 ile yuk testi + throughput/latency raporu
- Frontend operasyon paneli (`/admin`)
- Public URL + TLS mimarisi (`Ingress + cert-manager`)
- Ayri demo endpoint (`/demo/status`, `/demo/predict`)
- Ana sayfada terminalsiz "Try it" demo akisi (`/`)
- 7/24 operasyon runbook seti (incident, rollback, on-call, escalation)
- Ticari paketleme (Starter / Growth / Enterprise)

## Hizli Baslangic

```bash
make install
make demo-image
export ARTPULSE_API_KEY="guclu-bir-anahtar"
make serve ARTPULSE_API_KEY="$ARTPULSE_API_KEY"
```

## Rollout Kullanimi

Canary ornegi (trafik %10 challenger modele):

```bash
make serve \
  ARTPULSE_API_KEY="$ARTPULSE_API_KEY" \
  ROLLOUT_MODE=canary \
  CANARY_TRAFFIC_PERCENT=10 \
  CANDIDATE_MODEL_ALIAS=challenger
```

Kubernetes rollout kaydirma:

```bash
./scripts/rollout_shift.sh artpulse artpulse canary 25 blue
```

## Veri Kalitesi Katmani

Egitim oncesi otomatik kontroller:

- feature schema dogrulamasi
- NaN/inf kontrolu
- 0..1 aralik kontrolu
- minimum sample / sinif basi minimum sayi
- sinif dengesizligi uyarisi

Rapor dosyasi:

- `artifacts/data_quality_report.json`

## Frontend Operasyon Paneli

- URL: `http://localhost:8000/admin`
- Uretim URL: `https://api.wearetheartmakers.com/admin`
- Gosterilenler:
  - aktif model URI'lari
  - alias versiyonlari
  - rollout modu + trafik yuzdesi
  - request sayisi, hata orani, p95 latency
  - drift skoru + drift trend yonu
  - son retrain durumu + kalite gecis sonucu

Kurumsal login (SSO) icin ops endpoint mode:

- `OPS_OIDC_TRUST_HEADERS=true`
- `OPS_OIDC_ALLOWED_EMAIL_DOMAINS=wearetheartmakers.com`

## Yuk Testi

```bash
make loadtest K6_API_URL="http://host.docker.internal:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY"
make loadtest-report
```

Uretilen ciktilar:

- `loadtest/k6/k6-summary.json`
- `artifacts/loadtest_report.md`

## Canli Demo (Staging)

Lokal staging baz URL:

- API: `http://localhost:8000`
- Ops panel: `http://localhost:8000/admin`

Demo dosyalari:

- `docs/STAGING_DEMO.md`
- `examples/staging_predict_request.json`
- `examples/staging_predict_response.json`
- `examples/public_demo_request.json`
- `examples/public_demo_response.json`

Public demo endpoint (kisa omurlu token ile):

```bash
make serve ARTPULSE_API_KEY="$ARTPULSE_API_KEY" DEMO_ENABLED=true ARTPULSE_DEMO_KEY="signing-secret"
make demo-token API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
make demo-status API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
make demo-predict API_URL="http://localhost:8000" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
```

Public TLS smoke test:

```bash
make check-public-tls PUBLIC_API_URL="https://api.staging.<domain>" ARTPULSE_API_KEY="$ARTPULSE_API_KEY" ARTPULSE_DEMO_SUBJECT="portfolio-client"
```

Staging -> Production gecis ozet akisi:

1. Once staging (`api.staging.wearetheartmakers.com`) deploy + test
2. Staging'de smoke + kalite gate gecisi
3. Production promotion'u manuel onayla
4. Gate fail olursa rollback otomatik

1 dakikalik uptime + alarm kurulum rehberi:

- `docs/UPTIME_ALERTS_SETUP.md`

## Public URL + TLS Kurulum Referansi

- `docs/PUBLIC_URL_TLS_BLUEPRINT.md`
- `k8s/cert-manager-clusterissuer.yaml`
- `k8s/ingress-staging.yaml`
- `k8s/ingress-production.yaml`
- `k8s/ingress-demo-staging.yaml`
- `k8s/ingress-demo-production.yaml`
- `k8s/ingress-oauth2-staging.yaml`
- `k8s/ingress-oauth2-production.yaml`
- `k8s/network-policy.yaml`

## 7/24 Operasyon Runbook Seti

- `docs/runbooks/INCIDENT_RUNBOOK.md`
- `docs/runbooks/ROLLBACK_RUNBOOK.md`
- `docs/runbooks/ONCALL_RUNBOOK.md`
- `docs/runbooks/ESCALATION_MATRIX.md`

## Ticari Paketler ve SLA

| Paket | Kapsam | SLA |
| --- | --- | --- |
| Starter | Tek model pipeline, guvenli API, temel izleme | 99.0% uptime, mesai saati destek |
| Growth | Staging+prod, canary/blue-green, drift otomasyonu | 99.5% uptime, 8x5 destek, P1 <= 1 saat |
| Enterprise | 24/7 NOC/SRE, compliance, ileri seviye operasyon | 99.9% uptime, 24/7 destek, P1 <= 15 dk |

Detayli ticari kapsam:

- `docs/COMMERCIAL_PACKAGES.md`

## Kisisel Marketing Icın Konumlandirma

Musteri sunumunda su cizgiyi kullan:

- "Sadece model dogrulugu degil, production operasyon sorumlulugu da aliyorum."
- "Rollout, rollback, drift ve runbook sureclerini canli gosterebiliyorum."
- "Staging demo + KPI + SLA tablosu ile teknik teslimi ticari dilde aktarabiliyorum."

## Not

Komutlarin en guncel ve detayli aciklamasi icin her zaman `README.md` dosyasini referans al.
