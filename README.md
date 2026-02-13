# ArtPulse - Real Image MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

ArtPulse, bir ML modelini **egitim -> izleme -> model secimi -> servis -> container -> k8s deploy** akisiyla production'a tasimak icin hazirlanmis ornek bir MLOps projesidir.

Bu surumda artik sadece sentetik veri yok:

- Gercek image klasorlerinden feature extraction yapar.
- En iyi modeli MLflow Model Registry'ye kaydedip alias (`challenger/champion`) yonetebilir.
- API hem tabular feature hem de dogrudan image upload ile tahmin yapar.
- Drift raporu uretir ve periyodik retraining workflow'u vardir.

## 1) Ne Cozuluyor?

Bu repo su sorulara dogrudan cevap verir:

- "Gercek veriden model egitimi yaptin mi?"
- "Modeli registry alias ile promote edip production'a aldin mi?"
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
cd /Users/bg/MLflow/MLflow
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

- `/Users/bg/MLflow/MLflow/docs/GITHUB_SECURE_SETUP.md`

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

## 10) Onemli Dosyalar

```text
src/features.py               # synthetic + real image feature extraction
src/generate_image_dataset.py # local real-image style sample dataset uretici
src/train.py                  # train, compare, registry alias kayit
src/serve.py                  # API, image prediction, event logging
src/monitor_drift.py          # drift report uretimi
src/retrain_job.py            # periyodik retraining job
src/model_registry.py         # alias promotion yardimci komutlari
```

## 11) Dogrulama

```bash
make test
```

Testler sunlari kapsar:

- API health/readiness/predict/predict-image
- real image dataset extraction
- image dataset ile train akisi
