# ArtPulse - Turkce Dokumantasyon

Bu dosya, `README.md` iceriginin Turkce ozetidir.
Ana teknik dokumantasyonun guncel surumu `README.md` icindedir.

## Proje Ozeti

ArtPulse, bir AI/ML modelini production ortamina tasimak icin uçtan uca MLOps referansidir:

- Gercek image verisinden feature extraction + model egitimi
- MLflow tracking + Model Registry alias yonetimi
- FastAPI inference servisi
- Docker + Kubernetes deployment
- Drift monitoring + retraining otomasyonu

## Hizli Baslangic

```bash
cd /path/to/MLflow
make install
make demo-image
make serve
```

## Turkce Okuyucular Icin Onerilen Dosyalar

- `docs/CLIENT_PROPOSAL_TR.md` (musteri teklif metni)
- `docs/GITHUB_SECURE_SETUP.md` (guvenli GitHub secret/variable kurulumu)

## Projeye Eklenen Ek Hizmetler

Asagidaki basliklar proje kapsamina dahil edilmis roadmap ogeleridir:

- Yeni veri toplama altyapisi kurulumu
- Frontend urunlestirme (dashboard/UI katmani)
- Uzun sureli 7/24 operasyonel NOC/SRE hizmet modeli

## Kullanim Senaryosu (Kisisel Marketing)

Bu repo'yu firmalara sunarken su sekilde konumlandirabilirsin:

- "Model lifecycle ownership" (egitimden operasyon izlemeye kadar)
- Teknik teslimler: API, registry promotion, deployment automation
- Operasyonel teslimler: drift/gozlemleme/retraining sureci

## Not

Tum komutlarin en guncel hali icin her zaman `README.md` dosyasini referans al.
