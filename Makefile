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
API_URL?=http://localhost:8000
PUBLIC_API_URL?=https://api.staging.artpulse.example.com
ARTPULSE_API_KEY?=change-me-local-key
ARTPULSE_DEMO_KEY?=change-me-demo-key
ARTPULSE_DEMO_SUBJECT?=portfolio-client
DEMO_ENABLED?=true
K6_API_URL?=http://host.docker.internal:8000
ROLLOUT_MODE?=single
CANARY_TRAFFIC_PERCENT?=0
CANDIDATE_MODEL_ALIAS?=challenger
ACTIVE_COLOR?=blue
BLUE_GREEN_TRAFFIC_PERCENT?=0
DEPLOY_LEAD_TIME_MIN?=
ROLLBACK_TIME_MIN?=

.PHONY: help venv install train train-small generate-images train-images demo demo-image serve ui test predict predict-image demo-token demo-status demo-predict staging-demo check-public-tls monitor-drift retrain promote-alias monitoring-up monitoring-down metrics ops-summary loadtest loadtest-report business-impact-report docker-build docker-run k8s-apply k8s-apply-public-staging k8s-apply-public-production clean

help:
	@echo "Targets: venv install train train-small generate-images train-images demo demo-image serve ui test predict predict-image demo-token demo-status demo-predict staging-demo check-public-tls monitor-drift retrain promote-alias monitoring-up monitoring-down metrics ops-summary loadtest loadtest-report business-impact-report docker-build docker-run k8s-apply k8s-apply-public-staging k8s-apply-public-production clean"

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
	API_KEYS="$(ARTPULSE_API_KEY)" AUTH_REQUIRED=true \
	DEMO_ENABLED="$(DEMO_ENABLED)" \
	DEMO_JWK_CURRENT_KID="demo-v1" \
	DEMO_JWK_KEYS_JSON="{\"demo-v1\":\"$(ARTPULSE_DEMO_KEY)\"}" \
	DEMO_TOKEN_TTL_SECONDS=600 \
	SIEM_AUDIT_ENABLED=true \
	ROLLOUT_MODE="$(ROLLOUT_MODE)" CANARY_TRAFFIC_PERCENT="$(CANARY_TRAFFIC_PERCENT)" \
	CANDIDATE_MODEL_ALIAS="$(CANDIDATE_MODEL_ALIAS)" ACTIVE_COLOR="$(ACTIVE_COLOR)" \
	BLUE_GREEN_TRAFFIC_PERCENT="$(BLUE_GREEN_TRAFFIC_PERCENT)" \
	$(UVICORN) src.serve:app --host 0.0.0.0 --port 8000

ui:
	$(MLFLOW) ui --backend-store-uri ./mlruns --port 5000

test:
	$(PYTEST) -q

predict:
	curl -sS -X POST "$(API_URL)/predict" -H "Content-Type: application/json" -H "x-api-key: $(ARTPULSE_API_KEY)" \
	  -d '{"rows":[{"hue_mean":0.62,"sat_mean":0.55,"val_mean":0.70,"contrast":0.40,"edges":0.12}]}'

predict-image:
	IMG_B64=$$(base64 < $(DATASET_DIR)/vibrant/vibrant_0000.png | tr -d '\n'); \
	curl -sS -X POST "$(API_URL)/predict-image" -H "Content-Type: application/json" -H "x-api-key: $(ARTPULSE_API_KEY)" \
	  -d "{\"image_base64\":\"$$IMG_B64\"}"

demo-token:
	curl -sS -X POST "$(API_URL)/demo/token" \
	  -H "Content-Type: application/json" \
	  -H "x-api-key: $(ARTPULSE_API_KEY)" \
	  -d "{\"subject\":\"$(ARTPULSE_DEMO_SUBJECT)\",\"ttl_seconds\":600}" | python3 -m json.tool

demo-predict:
	DEMO_TOKEN=$$(curl -sS -X POST "$(API_URL)/demo/token" -H "Content-Type: application/json" -H "x-api-key: $(ARTPULSE_API_KEY)" -d "{\"subject\":\"$(ARTPULSE_DEMO_SUBJECT)\",\"ttl_seconds\":600}" | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"access_token\"])'); \
	curl -sS -X POST "$(API_URL)/demo/predict" \
	  -H "Content-Type: application/json" \
	  -H "Authorization: Bearer $$DEMO_TOKEN" \
	  --data-binary @examples/public_demo_request.json | python3 -m json.tool

demo-status:
	DEMO_TOKEN=$$(curl -sS -X POST "$(API_URL)/demo/token" -H "Content-Type: application/json" -H "x-api-key: $(ARTPULSE_API_KEY)" -d "{\"subject\":\"$(ARTPULSE_DEMO_SUBJECT)\",\"ttl_seconds\":600}" | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"access_token\"])'); \
	curl -sS "$(API_URL)/demo/status" -H "Authorization: Bearer $$DEMO_TOKEN" | python3 -m json.tool

staging-demo:
	curl -sS -X POST "$(API_URL)/predict" -H "Content-Type: application/json" -H "x-api-key: $(ARTPULSE_API_KEY)" \
	  --data-binary @examples/staging_predict_request.json

check-public-tls:
	./scripts/check_public_tls.sh "$(PUBLIC_API_URL)" "$(ARTPULSE_API_KEY)" "$(ARTPULSE_DEMO_SUBJECT)"

monitor-drift:
	$(PY) -m src.monitor_drift --training-summary artifacts/training_summary.json --prediction-log artifacts/prediction_events.jsonl --output artifacts/drift_report.json --history-output artifacts/drift_history.jsonl

retrain:
	$(PY) -m src.retrain_job --register-best --model-name $(MODEL_NAME) --model-alias challenger

promote-alias:
	$(PY) -m src.model_registry --tracking-uri "$${MLFLOW_TRACKING_URI}" --registry-uri "$${MLFLOW_REGISTRY_URI}" promote-alias --model-name $(MODEL_NAME) --source-alias challenger --target-alias champion

monitoring-up:
	docker compose -f observability/docker-compose.yml up -d

monitoring-down:
	docker compose -f observability/docker-compose.yml down

metrics:
	curl -sS "$(API_URL)/metrics" | sed -n '1,30p'

ops-summary:
	curl -sS "$(API_URL)/ops/summary" -H "x-api-key: $(ARTPULSE_API_KEY)" | python3 -m json.tool

loadtest:
	docker run --rm -i -e API_URL="$(K6_API_URL)" -e API_KEY="$(ARTPULSE_API_KEY)" \
	  -v "$$(pwd)/loadtest/k6:/scripts" grafana/k6:0.52.0 run /scripts/predict.js --summary-export=/scripts/k6-summary.json

loadtest-report:
	$(PY) -m src.loadtest_report --input loadtest/k6/k6-summary.json --output artifacts/loadtest_report.md

business-impact-report:
	$(PY) -m src.business_impact_report \
	  --training-summary artifacts/training_summary.json \
	  --drift-report artifacts/drift_report.json \
	  --loadtest-summary loadtest/k6/k6-summary.json \
	  --release-gate-report artifacts/release_gate_report.json \
	  --environment production \
	  --api-base-url "$(PUBLIC_API_URL)" \
	  --deploy-lead-time-min "$(DEPLOY_LEAD_TIME_MIN)" \
	  --rollback-time-min "$(ROLLBACK_TIME_MIN)" \
	  --output artifacts/business_impact_onepager.md

docker-build:
	docker build -t artpulse:latest .

docker-run:
	docker run --rm -p 8000:8000 artpulse:latest

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-apply-public-staging:
	kubectl apply -f k8s/cert-manager-clusterissuer.yaml
	kubectl apply -f k8s/oauth2-proxy-service.yaml
	kubectl apply -f k8s/oauth2-proxy-deployment.yaml
	kubectl apply -f k8s/ingress-oauth2-staging.yaml
	kubectl apply -f k8s/ingress-staging.yaml
	kubectl apply -f k8s/ingress-demo-staging.yaml
	kubectl apply -f k8s/network-policy.yaml

k8s-apply-public-production:
	kubectl apply -f k8s/cert-manager-clusterissuer.yaml
	kubectl apply -f k8s/oauth2-proxy-service.yaml
	kubectl apply -f k8s/oauth2-proxy-deployment.yaml
	kubectl apply -f k8s/ingress-oauth2-production.yaml
	kubectl apply -f k8s/ingress-production.yaml
	kubectl apply -f k8s/ingress-demo-production.yaml
	kubectl apply -f k8s/network-policy.yaml

clean:
	rm -rf mlruns artifacts .pytest_cache
