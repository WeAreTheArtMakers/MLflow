FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Bootstrap a tracked model during image build so the container starts ready.
RUN python -m src.train \
    --tracking-uri file:/app/mlruns \
    --experiment-name artpulse \
    --output-dir /app/artifacts

ENV MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_REGISTRY_URI= \
    MLFLOW_EXPERIMENT_NAME=artpulse \
    MODEL_URI_FILE=/app/artifacts/latest_model_uri.txt \
    USE_MODEL_REGISTRY_ALIAS=false \
    MODEL_NAME=artpulse-classifier \
    MODEL_ALIAS=champion \
    ROLLOUT_MODE=single \
    CANARY_TRAFFIC_PERCENT=0 \
    CANDIDATE_MODEL_ALIAS=challenger \
    ACTIVE_COLOR=blue \
    BLUE_MODEL_ALIAS=blue \
    GREEN_MODEL_ALIAS=green \
    BLUE_GREEN_TRAFFIC_PERCENT=0 \
    AUTH_REQUIRED=true \
    API_KEYS= \
    DEMO_ENABLED=false \
    DEMO_JWK_CURRENT_KID=demo-v1 \
    DEMO_JWK_KEYS_JSON= \
    DEMO_TOKEN_ISSUER=artpulse-demo \
    DEMO_TOKEN_AUDIENCE=artpulse-public-demo \
    DEMO_TOKEN_TTL_SECONDS=600 \
    DEMO_RATE_LIMIT_REQUESTS=30 \
    DEMO_RATE_LIMIT_WINDOW_SECONDS=60 \
    SIEM_AUDIT_ENABLED=false \
    SIEM_AUDIT_LOG_PATH=/app/artifacts/siem_audit_events.jsonl \
    JWT_SECRET= \
    JWT_ALGORITHM=HS256 \
    RATE_LIMIT_ENABLED=true \
    RATE_LIMIT_REQUESTS=120 \
    RATE_LIMIT_WINDOW_SECONDS=60 \
    PREDICTION_LOG_PATH=/app/artifacts/prediction_events.jsonl \
    TRAINING_SUMMARY_PATH=/app/artifacts/training_summary.json \
    DRIFT_REPORT_PATH=/app/artifacts/drift_report.json \
    PORT=8000

EXPOSE 8000
CMD ["bash", "-lc", "uvicorn src.serve:app --host 0.0.0.0 --port ${PORT}"]
