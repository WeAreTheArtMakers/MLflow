VENV=.venv
PYTHON=python3
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn
MLFLOW=$(VENV)/bin/mlflow
PYTEST=$(VENV)/bin/pytest

.PHONY: help venv install train train-small demo serve ui test predict docker-build docker-run k8s-apply clean

help:
	@echo "Targets: venv install train train-small demo serve ui test predict docker-build docker-run k8s-apply clean"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	$(PY) -m src.train

train-small:
	$(PY) -m src.train --n-samples 1500 --seed 7

demo:
	$(PY) -m src.demo --run-train --n-samples 3000 --seed 42

serve:
	$(UVICORN) src.serve:app --host 0.0.0.0 --port 8000

ui:
	$(MLFLOW) ui --backend-store-uri ./mlruns --port 5000

test:
	$(PYTEST) -q

predict:
	curl -sS -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
	  -d '{"rows":[{"hue_mean":0.62,"sat_mean":0.55,"val_mean":0.70,"contrast":0.40,"edges":0.12}]}'

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
