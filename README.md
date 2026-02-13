# ArtPulse - Real Image MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

## About

ArtPulse, bir AI/ML modelini fikirden production ortamina tasiyan uçtan uca bir MLOps referans projesidir.

Bu repo, firmalara sunulabilecek net bir teslim hikayesi verir:

- Gercek image verisinden feature extraction + model egitimi
- MLflow ile deney takibi, metrik loglama ve model artefact yonetimi
- Model Registry alias akisi (`challenger -> champion`) ile kontrollu model promotion
- FastAPI ile production-ready inference API (`/predict`, `/predict-image`)
- Docker + Kubernetes deployment
- Drift monitoring + planli retraining otomasyonu

## Kim Icin?

Bu proje su profillere hitap eder:

- "Modeli sadece egitmek degil, production'a almak" isteyen sirketler
- MLOps altyapisini hizli MVP olarak kurmak isteyen ekipler
- Teknik portfolio/teklif dosyasinda somut bir production ornegi gostermek isteyen bireysel uzmanlar

## Firmalara Sunumda Kisa Deger Onermesi

"ArtPulse ile bir modelin sadece dogruluk metriklerini degil, deployment, versiyonlama, rollback/promotion ve operasyonel izleme adimlarini da calisan bir akista teslim ediyorum."

## 1) Ne Cozuluyor?

Bu repo su sorulara dogrudan cevap verir:

- "Gercek veriden model egitimi yaptin mi?"
- "Modeli registry alias ile production'a promote ettin mi?"
- "CI/CD ile image build + deployment promotion yapiyor musun?"
- "Drift izleyip periyodik retraining yapiyor musun?"

Cevap: evet, bu repoda hepsi var.

## 2) Model Ozeti

Model, goruntuden cikarilan 5 ozet feature uzerinden sinif tahmini yapar:

- `hue_mean`
- `sat_mean`
- `val_mean`
- `contrast`
- `edges`

Etiketler:

- `minimal`
- `neo-pop`
- `surreal`
- `monochrome`
- `vibrant`

## 3) Hemen Basla

```bash
cd /path/to/MLflow
make install
```

### 3.1 Sentetik veri ile hizli demo

```bash
make demo
```

### 3.2 Gercek image pipeline ile demo

```bash
make demo-image
```

Bu komutlar:

- modeli egitir
- MLflow'a run/metric/artifact yazar
- en iyi modeli secer
- `artifacts/training_summary.json` ve `artifacts/example_predictions.json` uretir

## 4) Gercek Image Pipeline

### 4.1 Dataset formati

`examples/image_dataset_layout.txt` dosyasindaki klasor yapisini kullan.

Beklenen sinif klasorleri:

- `minimal`
- `neo-pop`
- `surreal`
- `monochrome`
- `vibrant`

### 4.2 Ornek dataset uret (local test icin)

```bash
make generate-images
```

Varsayilan cikti:

- `data/images/<label>/*.png`

### 4.3 Real image train calistir

```bash
make train-images
```

Manuel komut:

```bash
.venv/bin/python -m src.train \
  --dataset-type image \
  --dataset-dir data/images \
  --register-best \
  --model-name artpulse-classifier \
  --model-alias champion
```

## 5) Remote MLflow + Model Registry Alias

Remote ortam icin ornek env dosyasi:

- `examples/remote_env.example`

Ornek kullanim:

```bash
export MLFLOW_TRACKING_URI="http://mlflow.example.com"
export MLFLOW_REGISTRY_URI="http://mlflow.example.com"
export MODEL_NAME="artpulse-classifier"
export MODEL_ALIAS="champion"
```

### 5.1 Alias ile model yukleme (API)

```bash
export USE_MODEL_REGISTRY_ALIAS=true
make serve
```

Bu durumda API su URI'den model yukler:

- `models:/artpulse-classifier@champion`

### 5.2 Alias promotion

```bash
make promote-alias
```

Bu komut `challenger -> champion` promotion yapar.

## 6) API Kullanim

Servisi baslat:

```bash
make serve
```

Endpointler:

- `GET /health`
- `GET /ready`
- `POST /reload-model`
- `GET /metadata`
- `POST /predict` (tabular)
- `POST /predict-image` (base64 image payload)

Tabular tahmin:

```bash
make predict
```

Image tahmin (datasetten bir dosya ile):

```bash
make predict-image
```

Tahmin event log dosyasi:

- `artifacts/prediction_events.jsonl`

## 7) Drift Monitoring + Retraining

Drift raporu uret:

```bash
make monitor-drift
```

Rapor:

- `artifacts/drift_report.json`

Periyodik retraining komutu:

```bash
make retrain
```

Bu job en iyi modeli `challenger` alias'i ile kaydedebilir; sonra production promotion icin `make promote-alias` kullanilir.

## 8) CI/CD Workflows

`.github/workflows/` altinda:

- `ci.yml`: test + synthetic train + image train smoke
- `build-image.yml`: GHCR image build/push
- `deploy-promotion.yml`: model alias promotion + k8s image rollout
- `retrain.yml`: schedule/manual retrain pipeline

Guvenli GitHub secret/variable kurulumu icin:

- `docs/GITHUB_SECURE_SETUP.md`

## 9) Docker ve Kubernetes

Docker:

```bash
make docker-build
make docker-run
```

Kubernetes:

```bash
make k8s-apply
kubectl -n artpulse port-forward svc/artpulse 8000:80
curl -sS http://localhost:8000/health
```

`k8s/deployment.yaml` icinde registry alias env'leri hazirdir:

- `USE_MODEL_REGISTRY_ALIAS`
- `MODEL_NAME`
- `MODEL_ALIAS`

## 10) Bu Projeyi Kisisel Marketing Icin Nasil Konumlandirabilirsin?

Portfolio/CV/GitHub profilinde su sekilde konumlandir:

- Rol: "ML Engineer / MLOps Engineer"
- Odak: "Model lifecycle ownership (train -> registry -> deploy -> monitor)"
- Somut teslimler: "API, container, Kubernetes rollout, drift raporu, retraining workflow"

Onerilen 3 kisalik tanitim cümlesi:

1. "Real-image classification pipeline'i production-ready MLOps akisiyla teslim ettim."
2. "MLflow Model Registry alias stratejisiyle kontrollu model promotion kurdum."
3. "Drift monitoring ve retraining otomasyonuyla operasyonel sureklilik sagladim."

## 11) Firmalara Sunarken Gelistirme Onerileri (Roadmap)

### Faz 1 - Hemen ticari deger (1-2 hafta)

- Gercek musteri datasina bagli veri dogrulama kurallari
- API auth/rate-limit (JWT + gateway)
- SLO/SLI dashboard (latency, error rate, model freshness)

### Faz 2 - Kurumsal olceklendirme (2-4 hafta)

- Canary/A-B model rollout
- Feature store entegrasyonu
- Otomatik evaluation gate (promotion oncesi kalite esikleri)

### Faz 3 - Regulated / enterprise readiness (4+ hafta)

- Audit trail + lineage raporlama
- PII governance ve data retention policy
- On-prem / VPC deployment blueprint

## 12) Onemli Dosyalar

```text
src/features.py               # synthetic + real image feature extraction
src/generate_image_dataset.py # local real-image style sample dataset uretici
src/train.py                  # train, compare, registry alias kayit
src/serve.py                  # API, image prediction, event logging
src/monitor_drift.py          # drift report uretimi
src/retrain_job.py            # periyodik retraining job
src/model_registry.py         # alias promotion yardimci komutlari
```

## 13) Dogrulama

```bash
make test
```

Testler sunlari kapsar:

- API health/readiness/predict/predict-image
- real image dataset extraction
- image dataset ile train akisi
