# ArtPulse — Creative MLOps Demo (MLflow + FastAPI + Docker + Kubernetes)

ArtPulse is a small **end-to-end MLOps portfolio project** that demonstrates how to take an ML model from
**training → tracking (MLflow) → serving (FastAPI) → packaging (Docker) → deployment (Kubernetes)**.

## What does the model do?
ArtPulse predicts an **art mood / style label** from compact, interpretable features (e.g., color statistics,
contrast, saturation proxies, composition-like signals).

Example labels:
- `minimal`
- `neo-pop`
- `surreal`
- `monochrome`
- `vibrant`

> V1 uses synthetic data so the whole pipeline is reproducible.  
> Later, you can plug in real feature extraction from images.

---

## Quickstart (local)
### 1) Create venv + install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

