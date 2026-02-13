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
    PREDICTION_LOG_PATH=/app/artifacts/prediction_events.jsonl \
    PORT=8000

EXPOSE 8000
CMD ["bash", "-lc", "uvicorn src.serve:app --host 0.0.0.0 --port ${PORT}"]
