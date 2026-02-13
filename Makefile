VENV=.venv
PYTHON=python3
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn
MLFLOW=$(VENV)/bin/mlflow
PYTEST=$(VENV)/bin/pytest
DATASET_DIR?=data/images
MODEL_NAME?=artpulse-classifier
MODEL_ALIAS?=champion

.PHONY: help venv install train train-small generate-images train-images demo demo-image serve ui test predict predict-image monitor-drift retrain promote-alias docker-build docker-run k8s-apply clean

help:
	@echo "Targets: venv install train train-small generate-images train-images demo demo-image serve ui test predict predict-image monitor-drift retrain promote-alias docker-build docker-run k8s-apply clean"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	$(PY) -m src.train

train-small:
	$(PY) -m src.train --n-samples 1500 --seed 7

generate-images:
	$(PY) -m src.generate_image_dataset --output-dir $(DATASET_DIR) --images-per-label 120 --image-size 256 --seed 42

train-images:
	$(PY) -m src.train --dataset-type image --dataset-dir $(DATASET_DIR) --register-best --model-name $(MODEL_NAME) --model-alias $(MODEL_ALIAS)

demo:
	$(PY) -m src.demo --run-train --n-samples 3000 --seed 42

demo-image:
	$(PY) -m src.demo --run-train --dataset-type image --dataset-dir $(DATASET_DIR) --generate-sample-images

serve:
	$(UVICORN) src.serve:app --host 0.0.0.0 --port 8000

ui:
	$(MLFLOW) ui --backend-store-uri ./mlruns --port 5000

test:
	$(PYTEST) -q

predict:
	curl -sS -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
	  -d '{"rows":[{"hue_mean":0.62,"sat_mean":0.55,"val_mean":0.70,"contrast":0.40,"edges":0.12}]}'

predict-image:
	IMG_B64=$$(base64 < $(DATASET_DIR)/vibrant/vibrant_0000.png | tr -d '\n'); \
	curl -sS -X POST "http://localhost:8000/predict-image" -H "Content-Type: application/json" \
	  -d "{\"image_base64\":\"$$IMG_B64\"}"

monitor-drift:
	$(PY) -m src.monitor_drift --training-summary artifacts/training_summary.json --prediction-log artifacts/prediction_events.jsonl --output artifacts/drift_report.json

retrain:
	$(PY) -m src.retrain_job --register-best --model-name $(MODEL_NAME) --model-alias challenger

promote-alias:
	$(PY) -m src.model_registry --tracking-uri "$${MLFLOW_TRACKING_URI}" --registry-uri "$${MLFLOW_REGISTRY_URI}" promote-alias --model-name $(MODEL_NAME) --source-alias challenger --target-alias champion

docker-build:
	docker build -t artpulse:latest .

docker-run:
	docker run --rm -p 8000:8000 artpulse:latest

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

clean:
	rm -rf mlruns artifacts .pytest_cache
