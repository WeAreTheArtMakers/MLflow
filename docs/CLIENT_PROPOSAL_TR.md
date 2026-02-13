# Client Proposal - AI/ML Modeli Production'a Alma (ArtPulse Referansli)

## 1) Ozet

Merhaba [Musteri Adi],

Bu teklif, AI/ML modelinizi sadece egitmekle kalmayip production ortamina guvenli ve olceklendirilebilir sekilde almayi hedefler.
Calisma, ArtPulse referans mimarisi uzerinden ilerler:

- Gercek veri pipeline'i
- Deney takibi (MLflow)
- Model Registry ve kontrollu promotion (`challenger -> champion`)
- API servisleme (FastAPI)
- Containerization (Docker)
- Kubernetes deployment
- Drift monitoring + periyodik retraining

## 2) Is Hedefi

Bu proje sonunda ekibiniz su kazanimi elde eder:

- Model ciktilarinin tutarli ve izlenebilir sekilde production'da calismasi
- Model degisikliklerinin kontrollu rollout ve rollback yapabilmesi
- Operasyonel risklerin (drift, stale model, deployment hatalari) azaltilmasi
- MLOps surecinin dokumante ve tekrar edilebilir hale gelmesi

## 3) Kapsam (Scope)

### Dahil

- Veri/feature pipeline tasarimi (mevcut veri kaynaklarina bagli)
- Model egitimi ve benchmark (en az 2-3 model adayi)
- MLflow tracking + registry setup
- Inference API ve endpoint standardizasyonu
- Docker image uretimi
- Kubernetes deployment manifestleri
- CI/CD akislari (build, test, deploy promotion)
- Drift raporlama ve retraining job tasarimi
- Teknik dokumantasyon + handover
- Yeni veri toplama altyapisi kurulumu
- Frontend urunlestirme (dashboard / yonetim paneli)
- Uzun sureli 7/24 operasyonel NOC/SRE hizmet modeli (SLA bazli)

### Not

Kapsam 3 paket halinde sunulabilir:

- Temel paket (core MLOps teslimleri)
- Genisletilmis paket (data collection + frontend)
- Kurumsal operasyon paketi (7/24 NOC/SRE)

## 4) Teknik Teslimatlar

- Kaynak kod repo yapisi ve README
- Training pipeline scriptleri
- MLflow experiment ve model registry konfigrasyonu
- FastAPI inference servisi
- Dockerfile + image publishing workflow
- Kubernetes deployment/service/hpa dosyalari
- GitHub Actions workflow'lari
- Drift monitoring raporu ureten job
- Runbook: deploy, rollback, retrain, promote
- Data collection altyapi tasarimi ve entegrasyon notlari
- Frontend dashboard wireframe + MVP teslimi (pakete bagli)
- NOC/SRE operasyon runbook + alarm/escalation kurallari (pakete bagli)

## 5) Yol Haritasi ve Takvim

### Faz 1 - Assessment ve Baslangic (Hafta 1)

- Mevcut durum analizi
- Basari metriklerinin netlestirilmesi
- Hedef mimari ve guvenlik kontrol listesi

### Faz 2 - Build ve Integrasyon (Hafta 2-3)

- Pipeline + model training + registry
- API + Docker + Kubernetes
- CI/CD ilk canli akisi

### Faz 3 - Productization ve Operasyon (Hafta 4-6)

- Data collection altyapisi
- Frontend dashboard urunlestirme
- Drift monitoring + retraining operasyonu
- NOC/SRE operasyon setup (SLA, alarm, on-call)

## 6) Basari Kriterleri (Acceptance Criteria)

- Model API production ortaminda calisiyor ve health/readiness endpointleri dogru donuyor.
- Yeni model registry alias uzerinden promotion ile deploy edilebiliyor.
- Rollback proseduru test edilmis ve dokumante edilmis.
- Drift raporu en az bir test senaryosunda uretilmis.
- CI/CD pipeline kritik adimlarda green durumda.
- Data collection pipeline'dan en az bir uçtan uca veri akis test edilmis.
- Frontend dashboard'ta temel KPI'lar goruntulenebiliyor.
- NOC/SRE runbook onaylanmis ve alarm zinciri test edilmis.

## 7) Guvenlik ve Uyum Yaklasimi

- Environment bazli secret yonetimi (GitHub Environments)
- Least-privilege CI/CD permission modeli
- OIDC tabanli cloud erisimi (statik access key yerine)
- Production deploy icin manuel onay/reviewer kapisi
- Audit edilebilir deployment ve model version gecmisi

## 8) Musteriden Beklenenler

- Teknik muhatap (POC)
- Ortam erisimleri (staging/production)
- Veri erisimi ve veri sozlugu
- Guvenlik/politika gereksinimleri
- Onay akisi icin karar verici katilimi

## 9) Ticari Model (Ornek)

- Model A: Sabit kapsam + sabit fiyat
- Model B: Faz bazli (milestone) odeme
- Model C: Saatlik danismanlik + aylik retainer

Not: Nihai fiyatlandirma, veri karmasikligi, ortam sayisi ve SLA beklentisine gore netlestirilir.

## 10) Sonraki Adim

Onay durumunda asagidaki kickoff adimlariyla baslanir:

1. 60-90 dk teknik kickoff toplantisi
2. Kapsam/milestone kesinlestirme
3. Ilk hafta teslim planinin paylasimi

---

## Kisa E-posta Versiyonu (Copy/Paste)

Merhaba [Musteri Adi],

AI/ML modelinizi production'a alma ihtiyaciniz icin uçtan uca bir MLOps teslim modeli oneriyorum: model egitimi, MLflow tracking/registry, API servisleme, Docker/Kubernetes deploy, drift monitoring ve retraining otomasyonu.

Buna ek olarak veri toplama altyapisi, frontend urunlestirme ve 7/24 operasyonel NOC/SRE hizmetini de paketli olarak sunabiliyorum.

Uygunsa 60 dakikalik bir kickoff gorusmesiyle hedef metrikleri ve takvimi netlestirelim.

Tesekkurler,
[Ad Soyad]
[Unvan]
[Iletisim]
