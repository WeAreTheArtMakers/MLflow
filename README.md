# ArtPulse - Creative MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

ArtPulse, AI/ML modelini fikir asamasindan production'a tasima surecini gosteren ornek bir MLOps projesidir.

Bu repo ile su soruya net bir cevap verirsin:

"AI/ML modelini production ortaminda egittin, takip ettin, servis ettin ve deploy ettin mi?"

Bu projede cevap **evet**:

- Model adaylari egitilir ve karsilastirilir.
- Tum run'lar MLflow ile izlenir.
- En iyi model otomatik secilir (`deployment_ready=true`).
- FastAPI ile tahmin servisi acilir.
- Docker ve Kubernetes ile deploy edilir.

## 1) Proje ne ise yarar?

ArtPulse, goruntuye ait ozet feature'lardan sanat stili/mood etiketi tahmin eder.

Feature'lar:

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

Not: V1 surumu sentetik veri kullanir. Sonraki asamada gercek image pipeline eklenebilir.

## 2) Mimari akis

```text
train.py
  -> 3 model adayi egitimi
  -> MLflow run + metric + artifact loglama
  -> en iyi modeli secme
  -> artifacts/latest_model_uri.txt yazma

serve.py
  -> en iyi modeli bulup yukleme
  -> /health /ready /metadata /predict endpointleri

Dockerfile
  -> image build asamasinda model bootstrap egitimi

k8s/*.yaml
  -> Deployment + Service + HPA
```

## 3) Hizli baslangic (5-10 dakika)

### Kurulum

```bash
make install
```

### Tek komut demo (egitim + ornek tahmin)

```bash
make demo
```

Bu komut:

- 3 modeli egitir (`logistic_regression`, `random_forest`, `hist_gradient_boosting`)
- En iyi modeli secer
- Ornek satirlar uzerinde tahmin yapar
- Sonuclari `artifacts/example_predictions.json` icine yazar

### Son ornek calisma sonucu (2026-02-13, `--n-samples 2000 --seed 42`)

Asagidaki metrikler bu repoda alinmis guncel bir ornek ciktidir (run id her calistirmada degisir):

- `hist_gradient_boosting`: `f1_macro=0.9179`, `accuracy=0.9325` (best)
- `random_forest`: `f1_macro=0.9179`, `accuracy=0.9350`
- `logistic_regression`: `f1_macro=0.5418`, `accuracy=0.6725`

## 4) Model egitimi detaylari

Egitimi manuel calistirmak icin:

```bash
make train
```

Alternatif kucuk dataset ile hizli test:

```bash
make train-small
```

Uretilen dosyalar:

- `artifacts/latest_model_uri.txt`
- `artifacts/training_summary.json`

MLflow UI acmak icin:

```bash
make ui
# http://localhost:5000
```

## 5) API servisini kullanmak

Servisi baslat:

```bash
make serve
# http://localhost:8000
```

Endpointler:

- `GET /health`: servis ayakta mi, model yuklendi mi
- `GET /ready`: model hazir degilse 503 doner
- `GET /metadata`: label + feature sirasi
- `POST /predict`: tahmin

### Ornek API istegi

Istek ornegi dosyasi:

- `examples/predict_request.json`

Komut:

```bash
curl -sS -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

Ornek yanit dosyasi:

- `examples/predict_response.json`

## 6) Test

```bash
make test
```

Test kapsaminda:

- egitim pipeline smoke
- `/health` ve `/ready`
- gecerli/gecersiz payload kontrolu

## 7) Docker ile calistirma

```bash
make docker-build
make docker-run
```

Not:

- Docker image build sirasinda model bootstrap egitimi yapilir.
- Container acildiginda API modeli direkt bulup tahmine hazir olur.

## 8) Kubernetes deploy

```bash
make k8s-apply
kubectl -n artpulse port-forward svc/artpulse 8000:80
curl -sS http://localhost:8000/health
```

K8s tarafinda:

- `Deployment` readiness `/ready` endpointine bakar
- `Service` cluster icinde erisim verir
- `HPA` CPU bazli otomatik olcekleme yapar

## 9) Repo yapisi

```text
src/
  features.py      # sentetik veri + feature sozlesmesi
  train.py         # coklu model egitimi + MLflow loglama + best model secimi
  serve.py         # FastAPI inference servisi
  demo.py          # tek komut E2E demo
k8s/
  namespace.yaml
  deployment.yaml
  service.yaml
  hpa.yaml
examples/
  predict_request.json
  predict_response.json
tests/
  test_api.py
Dockerfile
Makefile
```

## 10) Production'a tasimak icin sonraki adimlar

- Sentetik veriyi gercek image feature pipeline ile degistir
- MLflow'u remote backend + Model Registry alias ile kullan
- CI/CD pipeline kur (test -> image -> deploy)
- Drift monitoring ve periyodik retraining ekle
