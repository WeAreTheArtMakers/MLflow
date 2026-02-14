import base64
import hashlib
import hmac
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.features import FEATURE_ORDER, LABELS, extract_image_features_from_bytes

app = FastAPI(title="ArtPulse API", version="0.5.0")


HTTP_LATENCY_BUCKETS: Tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 2.0)
METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class FeatureRow(BaseModel):
    hue_mean: float = Field(..., ge=0.0, le=1.0)
    sat_mean: float = Field(..., ge=0.0, le=1.0)
    val_mean: float = Field(..., ge=0.0, le=1.0)
    contrast: float = Field(..., ge=0.0, le=1.0)
    edges: float = Field(..., ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    rows: List[FeatureRow] = Field(..., min_length=1, description="Feature rows")


class PredictResponse(BaseModel):
    labels: List[str]
    predictions: List[int]
    mlflow_model_uri: str
    rollout_route: str
    rollout_mode: str


class PredictImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=16, description="Base64-encoded image content")


class PredictImageResponse(BaseModel):
    label: str
    prediction: int
    features: Dict[str, float]
    mlflow_model_uri: str
    rollout_route: str
    rollout_mode: str


class DemoPredictRequest(BaseModel):
    row: FeatureRow


class DemoPredictResponse(BaseModel):
    label: str
    prediction: int
    mlflow_model_uri: str
    rollout_route: str
    rollout_mode: str
    demo_request_id: str


class DemoTokenRequest(BaseModel):
    subject: str = Field(..., min_length=3, max_length=128)
    ttl_seconds: int = Field(default=600, ge=30, le=3600)


class DemoTokenResponse(BaseModel):
    token_type: str
    access_token: str
    expires_in: int
    issued_at: str
    expires_at: str
    kid: str
    audience: str
    issuer: str


_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_model_uri: str = ""
_load_error: str = ""
_load_lock = Lock()

_metrics_lock = Lock()
_http_requests_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
_http_latency_bucket_counts: Dict[Tuple[str, str, float], int] = defaultdict(int)
_http_latency_sum: Dict[Tuple[str, str], float] = defaultdict(float)
_http_latency_count: Dict[Tuple[str, str], int] = defaultdict(int)
_predictions_total: Dict[Tuple[str, str], int] = defaultdict(int)
_auth_failures_total: Dict[str, int] = defaultdict(int)
_rate_limit_exceeded_total: int = 0
_model_loaded: int = 0
_rollout_routed_total: Dict[Tuple[str, str, str], int] = defaultdict(int)

_rate_limit_lock = Lock()
_rate_limit_events: Dict[str, Deque[float]] = defaultdict(deque)
_demo_rate_limit_lock = Lock()
_demo_rate_limit_events: Dict[str, Deque[float]] = defaultdict(deque)
_audit_lock = Lock()

_secondary_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_secondary_model_uri: str = ""
_secondary_load_error: str = ""


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def _registry_uri() -> str:
    return os.getenv("MLFLOW_REGISTRY_URI", "")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "artpulse")


def _model_name() -> str:
    return os.getenv("MODEL_NAME", "artpulse-classifier")


def _model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "champion")


def _use_registry_alias() -> bool:
    return os.getenv("USE_MODEL_REGISTRY_ALIAS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prediction_log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG_PATH", "./artifacts/prediction_events.jsonl"))


def _image_feature_size() -> int:
    return int(os.getenv("IMAGE_FEATURE_SIZE", "128"))


def _auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _api_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "")


def _jwt_algorithm() -> str:
    return os.getenv("JWT_ALGORITHM", "HS256")


def _rate_limit_enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rate_limit_requests() -> int:
    return int(os.getenv("RATE_LIMIT_REQUESTS", "120"))


def _rate_limit_window_seconds() -> int:
    return int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


def _demo_enabled() -> bool:
    return os.getenv("DEMO_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _demo_jwk_current_kid() -> str:
    return os.getenv("DEMO_JWK_CURRENT_KID", "demo-v1").strip() or "demo-v1"


def _demo_jwk_keys() -> Dict[str, str]:
    raw_json = os.getenv("DEMO_JWK_KEYS_JSON", "").strip()
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            normalized: Dict[str, str] = {}
            for key, value in payload.items():
                key_str = str(key).strip()
                val_str = str(value).strip()
                if key_str and val_str:
                    normalized[key_str] = val_str
            if normalized:
                return normalized

    legacy_secret = os.getenv("DEMO_JWK_SECRET", "").strip()
    if legacy_secret:
        return {_demo_jwk_current_kid(): legacy_secret}

    return {}


def _demo_token_issuer() -> str:
    return os.getenv("DEMO_TOKEN_ISSUER", "artpulse-demo").strip() or "artpulse-demo"


def _demo_token_audience() -> str:
    return os.getenv("DEMO_TOKEN_AUDIENCE", "artpulse-public-demo").strip() or "artpulse-public-demo"


def _demo_rate_limit_requests() -> int:
    return int(os.getenv("DEMO_RATE_LIMIT_REQUESTS", "30"))


def _demo_rate_limit_window_seconds() -> int:
    return int(os.getenv("DEMO_RATE_LIMIT_WINDOW_SECONDS", "60"))


def _demo_token_ttl_seconds() -> int:
    raw = os.getenv("DEMO_TOKEN_TTL_SECONDS", "600")
    try:
        ttl = int(raw)
    except ValueError:
        ttl = 600
    return max(30, min(3600, ttl))


def _siem_audit_enabled() -> bool:
    return os.getenv("SIEM_AUDIT_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _siem_audit_log_path() -> Path:
    return Path(os.getenv("SIEM_AUDIT_LOG_PATH", "./artifacts/siem_audit_events.jsonl"))


def _rollout_mode() -> str:
    mode = os.getenv("ROLLOUT_MODE", "single").strip().lower()
    if mode not in {"single", "canary", "blue_green"}:
        return "single"
    return mode


def _canary_traffic_percent() -> float:
    raw = os.getenv("CANARY_TRAFFIC_PERCENT", "0")
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    return min(100.0, max(0.0, value))


def _candidate_model_alias() -> str:
    return os.getenv("CANDIDATE_MODEL_ALIAS", "challenger")


def _candidate_model_uri() -> str:
    return os.getenv("CANDIDATE_MODEL_URI", "").strip()


def _active_color() -> str:
    color = os.getenv("ACTIVE_COLOR", "blue").strip().lower()
    if color not in {"blue", "green"}:
        return "blue"
    return color


def _blue_model_alias() -> str:
    return os.getenv("BLUE_MODEL_ALIAS", "blue")


def _green_model_alias() -> str:
    return os.getenv("GREEN_MODEL_ALIAS", "green")


def _blue_model_uri() -> str:
    return os.getenv("BLUE_MODEL_URI", "").strip()


def _green_model_uri() -> str:
    return os.getenv("GREEN_MODEL_URI", "").strip()


def _blue_green_traffic_percent() -> float:
    raw = os.getenv("BLUE_GREEN_TRAFFIC_PERCENT", "0")
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    return min(100.0, max(0.0, value))


def _training_summary_path() -> Path:
    return Path(os.getenv("TRAINING_SUMMARY_PATH", "artifacts/training_summary.json"))


def _drift_report_path() -> Path:
    return Path(os.getenv("DRIFT_REPORT_PATH", "artifacts/drift_report.json"))


def _drift_history_path() -> Path:
    return Path(os.getenv("DRIFT_HISTORY_PATH", "artifacts/drift_history.jsonl"))


def _ops_oidc_trust_headers() -> bool:
    return os.getenv("OPS_OIDC_TRUST_HEADERS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ops_oidc_allowed_email_domains() -> set[str]:
    raw = os.getenv("OPS_OIDC_ALLOWED_EMAIL_DOMAINS", "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _secondary_traffic_percent() -> float:
    mode = _rollout_mode()
    if mode == "canary":
        return _canary_traffic_percent()
    if mode == "blue_green":
        return _blue_green_traffic_percent()
    return 0.0


def _set_model_loaded(value: int) -> None:
    global _model_loaded
    with _metrics_lock:
        _model_loaded = value


def _inc_http_request(method: str, path: str, status: int) -> None:
    with _metrics_lock:
        _http_requests_total[(method, path, str(status))] += 1


def _observe_http_latency(method: str, path: str, duration_seconds: float) -> None:
    with _metrics_lock:
        _http_latency_sum[(method, path)] += duration_seconds
        _http_latency_count[(method, path)] += 1
        for bucket in HTTP_LATENCY_BUCKETS:
            if duration_seconds <= bucket:
                _http_latency_bucket_counts[(method, path, bucket)] += 1


def _inc_prediction(source: str, label: str) -> None:
    with _metrics_lock:
        _predictions_total[(source, label)] += 1


def _inc_auth_failure(reason: str) -> None:
    with _metrics_lock:
        _auth_failures_total[reason] += 1


def _inc_rate_limit_exceeded() -> None:
    global _rate_limit_exceeded_total
    with _metrics_lock:
        _rate_limit_exceeded_total += 1


def _inc_rollout_route(source: str, mode: str, route: str) -> None:
    with _metrics_lock:
        _rollout_routed_total[(source, mode, route)] += 1


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")


def _labels(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{key}="{_escape_label_value(val)}"' for key, val in labels.items()]
    return "{" + ",".join(parts) + "}"


def _render_prometheus_metrics() -> str:
    lines: List[str] = []

    with _metrics_lock:
        lines.append("# HELP artpulse_http_requests_total Total HTTP requests")
        lines.append("# TYPE artpulse_http_requests_total counter")
        for (method, path, status), value in sorted(_http_requests_total.items()):
            label_str = _labels({"method": method, "path": path, "status": status})
            lines.append(f"artpulse_http_requests_total{label_str} {value}")

        lines.append("# HELP artpulse_http_request_latency_seconds HTTP request latency in seconds")
        lines.append("# TYPE artpulse_http_request_latency_seconds histogram")
        for method, path in sorted(_http_latency_count.keys()):
            for bucket in HTTP_LATENCY_BUCKETS:
                count = _http_latency_bucket_counts.get((method, path, bucket), 0)
                label_str = _labels({"method": method, "path": path, "le": f"{bucket:g}"})
                lines.append(f"artpulse_http_request_latency_seconds_bucket{label_str} {count}")
            total_count = _http_latency_count[(method, path)]
            inf_label_str = _labels({"method": method, "path": path, "le": "+Inf"})
            lines.append(f"artpulse_http_request_latency_seconds_bucket{inf_label_str} {total_count}")
            sum_label_str = _labels({"method": method, "path": path})
            lines.append(
                f"artpulse_http_request_latency_seconds_sum{sum_label_str} {_http_latency_sum[(method, path)]}"
            )
            lines.append(f"artpulse_http_request_latency_seconds_count{sum_label_str} {total_count}")

        lines.append("# HELP artpulse_predictions_total Total prediction count by source and label")
        lines.append("# TYPE artpulse_predictions_total counter")
        for (source, label), value in sorted(_predictions_total.items()):
            label_str = _labels({"source": source, "label": label})
            lines.append(f"artpulse_predictions_total{label_str} {value}")

        lines.append("# HELP artpulse_auth_failures_total Authentication failures")
        lines.append("# TYPE artpulse_auth_failures_total counter")
        for reason, value in sorted(_auth_failures_total.items()):
            label_str = _labels({"reason": reason})
            lines.append(f"artpulse_auth_failures_total{label_str} {value}")

        lines.append("# HELP artpulse_rate_limit_exceeded_total Rate-limit exceeded events")
        lines.append("# TYPE artpulse_rate_limit_exceeded_total counter")
        lines.append(f"artpulse_rate_limit_exceeded_total {_rate_limit_exceeded_total}")

        lines.append(
            "# HELP artpulse_rollout_routed_total Requests routed by rollout mode and route"
        )
        lines.append("# TYPE artpulse_rollout_routed_total counter")
        for (source, mode, route), value in sorted(_rollout_routed_total.items()):
            label_str = _labels({"source": source, "mode": mode, "route": route})
            lines.append(f"artpulse_rollout_routed_total{label_str} {value}")

        lines.append("# HELP artpulse_model_loaded Whether a model is loaded (1 or 0)")
        lines.append("# TYPE artpulse_model_loaded gauge")
        lines.append(f"artpulse_model_loaded {_model_loaded}")

    return "\n".join(lines) + "\n"


def _histogram_percentile_ms(bucket_counts: Dict[float, int], total_count: int, percentile: float) -> float:
    if total_count <= 0:
        return 0.0
    target = total_count * percentile
    for bucket in HTTP_LATENCY_BUCKETS:
        if bucket_counts.get(bucket, 0) >= target:
            return round(bucket * 1000.0, 2)
    return round(HTTP_LATENCY_BUCKETS[-1] * 1000.0, 2)


def _build_runtime_kpis() -> Dict[str, object]:
    inference_paths = {"/predict", "/predict-image", "/demo/predict"}
    all_buckets = {bucket: 0 for bucket in HTTP_LATENCY_BUCKETS}
    inference_buckets = {bucket: 0 for bucket in HTTP_LATENCY_BUCKETS}

    with _metrics_lock:
        total_requests = sum(_http_requests_total.values())
        error_requests = sum(
            value for (_, _, status), value in _http_requests_total.items() if status.startswith("5")
        )
        prediction_count = sum(_predictions_total.values())
        auth_failures = sum(_auth_failures_total.values())

        latency_count = sum(_http_latency_count.values())
        latency_sum = sum(_http_latency_sum.values())

        inference_latency_count = 0
        inference_latency_sum = 0.0
        for (method, path), count in _http_latency_count.items():
            if path in inference_paths:
                inference_latency_count += count
                inference_latency_sum += _http_latency_sum.get((method, path), 0.0)

        for (_, path, bucket), count in _http_latency_bucket_counts.items():
            all_buckets[bucket] += count
            if path in inference_paths:
                inference_buckets[bucket] += count

    error_rate = (error_requests / total_requests) if total_requests else 0.0
    avg_latency_ms = (latency_sum / latency_count * 1000.0) if latency_count else 0.0
    inference_avg_latency_ms = (
        inference_latency_sum / inference_latency_count * 1000.0 if inference_latency_count else 0.0
    )

    return {
        "request_count": int(total_requests),
        "prediction_count": int(prediction_count),
        "error_count": int(error_requests),
        "error_rate": round(error_rate, 6),
        "error_rate_percent": round(error_rate * 100.0, 3),
        "p95_latency_ms": _histogram_percentile_ms(all_buckets, latency_count, 0.95),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "inference_request_count": int(inference_latency_count),
        "inference_p95_latency_ms": _histogram_percentile_ms(
            inference_buckets, inference_latency_count, 0.95
        ),
        "inference_avg_latency_ms": round(inference_avg_latency_ms, 2),
        "auth_failures": int(auth_failures),
        "rate_limit_exceeded_total": int(_rate_limit_exceeded_total),
    }


def _b64url_decode(encoded: str) -> bytes:
    padded = encoded + ("=" * (-len(encoded) % 4))
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _encode_jwt_hs256(header: Dict[str, object], payload: Dict[str, object], secret: str) -> str:
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def _decode_jwt_parts(token: str) -> Tuple[Dict[str, object], Dict[str, object], bytes, bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT token")

    header_b64, payload_b64, signature_b64 = parts
    try:
        header_obj = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        payload_obj = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        signature = _b64url_decode(signature_b64)
    except Exception as exc:
        raise ValueError("Malformed JWT payload") from exc

    if not isinstance(header_obj, dict) or not isinstance(payload_obj, dict):
        raise ValueError("Invalid JWT structure")

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    return header_obj, payload_obj, signature, signing_input


def _decode_and_verify_jwt(token: str, secret: str, algorithm: str) -> Dict[str, object]:
    if algorithm.upper() != "HS256":
        raise ValueError("Only HS256 JWT algorithm is supported")

    header_obj, payload_obj, signature, signing_input = _decode_jwt_parts(token)
    if header_obj.get("alg") != "HS256":
        raise ValueError("JWT algorithm header mismatch")

    expected_signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("JWT signature verification failed")

    now = int(time.time())
    exp_claim = payload_obj.get("exp")
    if exp_claim is not None and now >= int(exp_claim):
        raise ValueError("JWT token has expired")

    nbf_claim = payload_obj.get("nbf")
    if nbf_claim is not None and now < int(nbf_claim):
        raise ValueError("JWT token is not active yet")

    return payload_obj


def _verify_demo_token(token: str) -> str:
    keys = _demo_jwk_keys()
    if not keys:
        raise ValueError("Demo JWK keys are not configured")

    header_obj, payload_obj, signature, signing_input = _decode_jwt_parts(token)
    if str(header_obj.get("alg", "")).upper() != "HS256":
        raise ValueError("Demo token algorithm must be HS256")

    kid = str(header_obj.get("kid", "")).strip()
    if not kid:
        raise ValueError("Demo token missing kid")

    secret = keys.get(kid)
    if not secret:
        raise ValueError("Demo token kid is not recognized")

    expected_signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Demo token signature verification failed")

    now = int(time.time())
    exp_claim = payload_obj.get("exp")
    if exp_claim is None or now >= int(exp_claim):
        raise ValueError("Demo token has expired")

    nbf_claim = payload_obj.get("nbf")
    if nbf_claim is not None and now < int(nbf_claim):
        raise ValueError("Demo token is not active yet")

    iss_claim = str(payload_obj.get("iss", ""))
    if iss_claim != _demo_token_issuer():
        raise ValueError("Demo token issuer mismatch")

    expected_aud = _demo_token_audience()
    aud_claim = payload_obj.get("aud")
    if isinstance(aud_claim, list):
        if expected_aud not in [str(item) for item in aud_claim]:
            raise ValueError("Demo token audience mismatch")
    elif str(aud_claim) != expected_aud:
        raise ValueError("Demo token audience mismatch")

    subject = str(payload_obj.get("sub", "")).strip()
    if not subject:
        raise ValueError("Demo token has no subject")
    return subject


def _mint_demo_token(subject: str, ttl_seconds: int) -> Tuple[str, int, int, str]:
    kid = _demo_jwk_current_kid()
    keys = _demo_jwk_keys()
    secret = keys.get(kid)
    if not secret:
        raise ValueError(f"Active demo kid '{kid}' not found in key set")

    now = int(time.time())
    ttl = max(30, min(3600, int(ttl_seconds)))
    exp = now + ttl
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    payload = {
        "sub": subject,
        "iss": _demo_token_issuer(),
        "aud": _demo_token_audience(),
        "iat": now,
        "nbf": now,
        "exp": exp,
    }
    token = _encode_jwt_hs256(header=header, payload=payload, secret=secret)
    return token, now, exp, kid


def _apply_mlflow_uris() -> None:
    mlflow.set_tracking_uri(_tracking_uri())
    if _registry_uri():
        mlflow.set_registry_uri(_registry_uri())


def _discover_model_uri() -> str:
    env_model_uri = os.getenv("MODEL_URI", "").strip()
    if env_model_uri:
        return env_model_uri

    if _use_registry_alias():
        return f"models:/{_model_name()}@{_model_alias()}"

    uri_file = Path(os.getenv("MODEL_URI_FILE", "./artifacts/latest_model_uri.txt"))
    if uri_file.exists():
        model_uri = uri_file.read_text(encoding="utf-8").strip()
        if model_uri:
            return model_uri

    client = MlflowClient(tracking_uri=_tracking_uri(), registry_uri=_registry_uri() or None)
    exp = client.get_experiment_by_name(_experiment_name())
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{_experiment_name()}' not found. Train first with python3 -m src.train"
        )

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string='tags.deployment_ready = "true"',
        order_by=["metrics.f1_macro DESC", "attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["metrics.f1_macro DESC", "attributes.start_time DESC"],
            max_results=1,
        )
    if not runs:
        raise RuntimeError(
            f"No MLflow runs found in experiment '{_experiment_name()}'. Train first with python3 -m src.train"
        )

    return f"runs:/{runs[0].info.run_id}/model"


def _alias_model_uri(alias: str) -> str:
    return f"models:/{_model_name()}@{alias}"


def _resolve_rollout_model_uris() -> Tuple[str, str]:
    primary_uri = _discover_model_uri()
    mode = _rollout_mode()

    if mode == "single":
        return primary_uri, ""

    if mode == "canary":
        secondary_uri = _candidate_model_uri() or _alias_model_uri(_candidate_model_alias())
        return primary_uri, secondary_uri

    # blue_green
    blue_uri = _blue_model_uri() or _alias_model_uri(_blue_model_alias())
    green_uri = _green_model_uri() or _alias_model_uri(_green_model_alias())
    if _active_color() == "blue":
        return blue_uri, green_uri
    return green_uri, blue_uri


def _consume_sliding_window(
    identity: str,
    limit: int,
    window_seconds: int,
    lock: Lock,
    event_store: Dict[str, Deque[float]],
) -> Tuple[bool, float]:
    now = time.monotonic()

    with lock:
        events = event_store[identity]
        while events and (now - events[0]) > window_seconds:
            events.popleft()

        if len(events) >= limit:
            retry_after = max(0.0, window_seconds - (now - events[0]))
            return False, retry_after

        events.append(now)

    return True, 0.0


def _consume_rate_limit(identity: str) -> Tuple[bool, float]:
    if not _rate_limit_enabled():
        return True, 0.0

    return _consume_sliding_window(
        identity=identity,
        limit=_rate_limit_requests(),
        window_seconds=_rate_limit_window_seconds(),
        lock=_rate_limit_lock,
        event_store=_rate_limit_events,
    )


def _routing_bucket(identity: str, routing_key: str) -> float:
    seed = f"{identity}:{routing_key}".encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()
    value = int(digest[:8], 16) % 10000
    return value / 100.0


def _should_use_secondary(identity: str, routing_key: str) -> bool:
    if _secondary_model is None:
        return False

    percent = _secondary_traffic_percent()
    if percent <= 0:
        return False
    if percent >= 100:
        return True

    return _routing_bucket(identity, routing_key) < percent


def _select_inference_model(identity: str, routing_key: str) -> Tuple[mlflow.pyfunc.PyFuncModel, str, str]:
    if _model is None:
        raise RuntimeError("Primary model is not loaded")

    if _should_use_secondary(identity=identity, routing_key=routing_key):
        if _secondary_model is None:
            return _model, _model_uri, "primary"
        return _secondary_model, _secondary_model_uri, "secondary"

    return _model, _model_uri, "primary"


def _authenticate_identity(authorization: Optional[str], x_api_key: Optional[str]) -> str:
    if not _auth_required():
        return "anonymous"

    if x_api_key:
        keys = _api_keys()
        if keys and x_api_key in keys:
            return f"api_key:{x_api_key[:6]}"
        if not keys:
            _inc_auth_failure("api_key_not_configured")
            raise HTTPException(status_code=401, detail="API key auth is not configured")
        _inc_auth_failure("invalid_api_key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        secret = _jwt_secret()
        if not secret:
            _inc_auth_failure("jwt_not_configured")
            raise HTTPException(status_code=401, detail="JWT auth is not configured")
        try:
            payload = _decode_and_verify_jwt(token, secret, _jwt_algorithm())
        except Exception as exc:
            _inc_auth_failure("invalid_jwt")
            raise HTTPException(status_code=401, detail=f"Invalid JWT token: {exc}") from exc

        subject = str(payload.get("sub") or payload.get("user_id") or payload.get("email") or "")
        if not subject:
            _inc_auth_failure("jwt_missing_subject")
            raise HTTPException(status_code=401, detail="JWT token has no subject")
        return f"jwt:{subject}"

    _inc_auth_failure("missing_credentials")
    raise HTTPException(status_code=401, detail="Missing credentials")


def _apply_rate_limit(identity: str) -> str:
    allowed, retry_after = _consume_rate_limit(identity=identity)
    if not allowed:
        _inc_rate_limit_exceeded()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(max(1, int(round(retry_after))))},
        )
    return identity


def _proxy_oidc_identity(
    x_auth_request_email: Optional[str],
    x_auth_request_user: Optional[str],
    x_forwarded_user: Optional[str],
    x_user: Optional[str],
) -> str:
    candidates = [
        (x_auth_request_email or "").strip(),
        (x_auth_request_user or "").strip(),
        (x_forwarded_user or "").strip(),
        (x_user or "").strip(),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _is_allowed_oidc_email(identity: str) -> bool:
    allowed_domains = _ops_oidc_allowed_email_domains()
    if not allowed_domains:
        return True
    if "@" not in identity:
        return False
    domain = identity.split("@", 1)[1].lower()
    return domain in allowed_domains


def require_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> str:
    identity = _authenticate_identity(authorization=authorization, x_api_key=x_api_key)
    return _apply_rate_limit(identity=identity)


def require_ops_access(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    x_auth_request_email: Optional[str] = Header(default=None),
    x_auth_request_user: Optional[str] = Header(default=None),
    x_forwarded_user: Optional[str] = Header(default=None),
    x_user: Optional[str] = Header(default=None),
) -> str:
    if _ops_oidc_trust_headers():
        proxy_identity = _proxy_oidc_identity(
            x_auth_request_email=x_auth_request_email,
            x_auth_request_user=x_auth_request_user,
            x_forwarded_user=x_forwarded_user,
            x_user=x_user,
        )
        if proxy_identity:
            if not _is_allowed_oidc_email(proxy_identity):
                _inc_auth_failure("ops_oidc_forbidden_domain")
                raise HTTPException(status_code=403, detail="SSO account is not in allowed domains")
            return _apply_rate_limit(identity=f"oidc:{proxy_identity}")

    identity = _authenticate_identity(authorization=authorization, x_api_key=x_api_key)
    return _apply_rate_limit(identity=identity)


def _consume_demo_rate_limit(identity: str) -> Tuple[bool, float]:
    return _consume_sliding_window(
        identity=identity,
        limit=_demo_rate_limit_requests(),
        window_seconds=_demo_rate_limit_window_seconds(),
        lock=_demo_rate_limit_lock,
        event_store=_demo_rate_limit_events,
    )


def require_demo_access(
    authorization: Optional[str] = Header(default=None),
    x_demo_token: Optional[str] = Header(default=None),
) -> str:
    if not _demo_enabled():
        _inc_auth_failure("demo_disabled")
        raise HTTPException(status_code=404, detail="Not found")

    keys = _demo_jwk_keys()
    if not keys:
        _inc_auth_failure("demo_not_configured")
        raise HTTPException(status_code=503, detail="Demo endpoint is not configured")

    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    elif x_demo_token:
        token = x_demo_token.strip()

    if not token:
        _inc_auth_failure("demo_missing_token")
        raise HTTPException(status_code=401, detail="Missing demo token")

    try:
        subject = _verify_demo_token(token)
    except Exception as exc:
        _inc_auth_failure("demo_invalid_token")
        raise HTTPException(status_code=401, detail=f"Invalid demo token: {exc}") from exc

    identity = f"demo_token:{subject}"
    allowed, retry_after = _consume_demo_rate_limit(identity=identity)
    if not allowed:
        _inc_rate_limit_exceeded()
        raise HTTPException(
            status_code=429,
            detail="Demo rate limit exceeded",
            headers={"Retry-After": str(max(1, int(round(retry_after))))},
        )

    return identity


def _append_prediction_events(
    rows: List[Dict[str, float]],
    prediction_ids: List[int],
    labels: List[str],
    source: str,
    actor: str,
    model_uri: str,
    rollout_route: str,
    rollout_mode: str,
) -> None:
    if not rows:
        return

    log_path = _prediction_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        for row, pred_id, label in zip(rows, prediction_ids, labels):
            event = {
                "timestamp": now,
                "source": source,
                "actor": actor,
                "model_uri": model_uri,
                "rollout_route": rollout_route,
                "rollout_mode": rollout_mode,
                "features": row,
                "prediction": int(pred_id),
                "label": label,
            }
            fh.write(json.dumps(event) + "\n")


def _append_audit_event(event: Dict[str, object]) -> None:
    if not _siem_audit_enabled():
        return
    try:
        audit_path = _siem_audit_log_path()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with _audit_lock:
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
    except Exception:
        # Audit trail should not block serving traffic.
        return


def load_model(force_reload: bool = False) -> None:
    global _model, _model_uri, _load_error, _secondary_model, _secondary_model_uri, _secondary_load_error
    with _load_lock:
        if _model is not None and not force_reload:
            return

        _apply_mlflow_uris()

        primary_uri, secondary_uri = _resolve_rollout_model_uris()
        primary_loaded = mlflow.pyfunc.load_model(primary_uri)

        secondary_loaded = None
        secondary_error = ""
        if secondary_uri:
            try:
                secondary_loaded = mlflow.pyfunc.load_model(secondary_uri)
            except Exception as exc:
                secondary_error = str(exc)

        _model = primary_loaded
        _model_uri = primary_uri
        _secondary_model = secondary_loaded
        _secondary_model_uri = secondary_uri if secondary_loaded is not None else ""
        _secondary_load_error = secondary_error

        if secondary_error:
            _load_error = f"Secondary model load failed: {secondary_error}"
        else:
            _load_error = ""
        _set_model_loaded(1 if _model is not None else 0)


def _read_json_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, max_items: int = 60) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    items: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                items.append(payload)
    if max_items > 0 and len(items) > max_items:
        return items[-max_items:]
    return items


def _build_drift_trend(history_items: List[Dict[str, object]]) -> Dict[str, object]:
    points: List[Dict[str, object]] = []
    for item in history_items:
        try:
            score = float(item.get("drift_score", 0.0))
        except Exception:
            score = 0.0
        points.append(
            {
                "generated_at": str(item.get("generated_at", "unknown")),
                "drift_score": round(score, 6),
                "drift_detected": bool(item.get("drift_detected", False)),
            }
        )

    points = points[-12:]
    direction = "flat"
    delta = 0.0
    if len(points) >= 2:
        recent = points[-3:]
        previous = points[-6:-3] if len(points) >= 6 else points[:-1]
        if previous:
            recent_avg = sum(float(p["drift_score"]) for p in recent) / len(recent)
            prev_avg = sum(float(p["drift_score"]) for p in previous) / len(previous)
            delta = recent_avg - prev_avg
            if delta > 0.05:
                direction = "up"
            elif delta < -0.05:
                direction = "down"

    return {
        "direction": direction,
        "delta": round(delta, 6),
        "points": points,
    }


def _header_first(request: Request, header_name: str) -> str:
    raw = request.headers.get(header_name, "")
    if not raw:
        return ""
    return raw.split(",")[0].strip()


def _external_base_url(request: Request) -> str:
    scheme = _header_first(request, "x-forwarded-proto") or request.url.scheme
    host = _header_first(request, "x-forwarded-host") or request.headers.get("host", "")
    if not host:
        host = request.url.netloc

    forwarded_port = _header_first(request, "x-forwarded-port")
    if forwarded_port and host and ":" not in host:
        if (scheme == "https" and forwarded_port != "443") or (scheme == "http" and forwarded_port != "80"):
            host = f"{host}:{forwarded_port}"

    prefix = _header_first(request, "x-forwarded-prefix")
    prefix_part = f"/{prefix.strip('/')}" if prefix else ""
    return f"{scheme}://{host}{prefix_part}".rstrip("/")


def _get_alias_version(alias: str) -> str:
    if not _registry_uri():
        return "registry-disabled"
    try:
        client = MlflowClient(tracking_uri=_tracking_uri(), registry_uri=_registry_uri() or None)
        mv = client.get_model_version_by_alias(name=_model_name(), alias=alias)
    except Exception:
        return "not-set"
    return str(mv.version)


def _build_ops_summary() -> Dict[str, object]:
    training_summary = _read_json_file(_training_summary_path())
    drift_report = _read_json_file(_drift_report_path())
    drift_history = _read_jsonl(_drift_history_path(), max_items=60)
    rollout_mode = _rollout_mode()
    runtime_kpis = _build_runtime_kpis()

    last_retrain_at = training_summary.get("generated_at", "unknown")
    best_model_uri = str(training_summary.get("best_model_uri", "") or "")
    retrain_success = bool(best_model_uri)
    data_quality_payload = training_summary.get("data_quality", {})
    data_quality_passed = None
    if isinstance(data_quality_payload, dict) and "passed" in data_quality_payload:
        data_quality_passed = bool(data_quality_payload.get("passed"))

    alias_versions: Dict[str, str] = {
        "champion": _get_alias_version("champion"),
        "challenger": _get_alias_version("challenger"),
    }
    if rollout_mode == "blue_green":
        alias_versions["blue"] = _get_alias_version(_blue_model_alias())
        alias_versions["green"] = _get_alias_version(_green_model_alias())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "service": {"name": app.title, "version": app.version},
        "model": {
            "primary_uri": _model_uri,
            "secondary_uri": _secondary_model_uri,
            "model_name": _model_name(),
            "registry_enabled": bool(_registry_uri()),
            "registry_uri": _registry_uri() or "not-configured",
            "registry_alias_mode_enabled": _use_registry_alias(),
            "alias_versions": alias_versions,
            "last_retrain_at": last_retrain_at,
            "last_retrain_success": retrain_success,
            "last_retrain_model_uri": best_model_uri or "unknown",
            "data_quality_passed": data_quality_passed,
        },
        "drift": {
            "drift_detected": bool(drift_report.get("drift_detected", False)),
            "drift_score": float(drift_report.get("drift_score", 0.0)),
            "event_count": int(drift_report.get("event_count", 0))
            if drift_report.get("event_count") is not None
            else 0,
            "generated_at": drift_report.get("generated_at", "unknown"),
            "trend": _build_drift_trend(drift_history),
        },
        "rollout": {
            "mode": rollout_mode,
            "secondary_percent": _secondary_traffic_percent(),
            "primary_route": "blue" if rollout_mode == "blue_green" and _active_color() == "blue" else (
                "green" if rollout_mode == "blue_green" else "primary"
            ),
            "secondary_route": "green"
            if rollout_mode == "blue_green" and _active_color() == "blue"
            else ("blue" if rollout_mode == "blue_green" else "secondary"),
        },
        "kpis": runtime_kpis,
        "retrain": {
            "status": "success" if retrain_success else ("unknown" if not training_summary else "failed"),
            "last_retrain_at": last_retrain_at,
            "last_model_uri": best_model_uri or "unknown",
            "data_quality_passed": data_quality_passed,
            "candidate_count": len(training_summary.get("candidates", []))
            if isinstance(training_summary.get("candidates"), list)
            else 0,
        },
        "paths": {
            "training_summary": str(_training_summary_path()),
            "drift_report": str(_drift_report_path()),
            "drift_history": str(_drift_history_path()),
            "prediction_log": str(_prediction_log_path()),
        },
    }


@app.middleware("http")
async def capture_http_metrics(request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start
        _inc_http_request(method=method, path=path, status=status_code)
        _observe_http_latency(method=method, path=path, duration_seconds=duration)

        if path.startswith("/admin") or path.startswith("/ops/summary"):
            forwarded_for = request.headers.get("x-forwarded-for", "")
            source_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
                request.client.host if request.client else "unknown"
            )
            actor = (
                request.headers.get("x-auth-request-email")
                or request.headers.get("x-forwarded-user")
                or request.headers.get("x-user")
                or "unknown"
            )
            _append_audit_event(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "path": path,
                    "method": method,
                    "status": status_code,
                    "actor": actor,
                    "source_ip": source_ip,
                    "request_id": request.headers.get("x-request-id", ""),
                    "message": "admin_access_audit",
                }
            )


@app.on_event("startup")
def on_startup() -> None:
    try:
        load_model()
    except Exception as exc:
        global _load_error, _secondary_model, _secondary_model_uri, _secondary_load_error
        _load_error = str(exc)
        _secondary_model = None
        _secondary_model_uri = ""
        _secondary_load_error = ""
        _set_model_loaded(0)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    _set_model_loaded(1 if _model is not None else 0)
    return PlainTextResponse(_render_prometheus_metrics(), media_type=METRICS_CONTENT_TYPE)


@app.get("/health")
def health() -> dict:
    mode = _rollout_mode()
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "secondary_model_loaded": _secondary_model is not None,
        "mlflow_tracking_uri": _tracking_uri(),
        "mlflow_registry_uri": _registry_uri(),
        "mlflow_experiment": _experiment_name(),
        "mlflow_model_uri": _model_uri,
        "secondary_model_uri": _secondary_model_uri,
        "model_registry": {
            "enabled": _use_registry_alias(),
            "model_name": _model_name(),
            "model_alias": _model_alias(),
        },
        "rollout": {
            "mode": mode,
            "secondary_percent": _secondary_traffic_percent(),
            "candidate_alias": _candidate_model_alias(),
            "active_color": _active_color(),
            "blue_alias": _blue_model_alias(),
            "green_alias": _green_model_alias(),
        },
        "auth": {
            "required": _auth_required(),
            "api_keys_configured": len(_api_keys()),
            "jwt_enabled": bool(_jwt_secret()),
            "rate_limit_enabled": _rate_limit_enabled(),
            "rate_limit_requests": _rate_limit_requests(),
            "rate_limit_window_seconds": _rate_limit_window_seconds(),
            "ops_oidc_enabled": _ops_oidc_trust_headers(),
            "ops_oidc_allowed_email_domains": sorted(_ops_oidc_allowed_email_domains()),
        },
        "demo": {
            "enabled": _demo_enabled(),
            "jwk_keys_configured": len(_demo_jwk_keys()),
            "active_kid": _demo_jwk_current_kid(),
            "token_issuer": _demo_token_issuer(),
            "token_audience": _demo_token_audience(),
            "token_ttl_seconds": _demo_token_ttl_seconds(),
            "rate_limit_requests": _demo_rate_limit_requests(),
            "rate_limit_window_seconds": _demo_rate_limit_window_seconds(),
        },
        "siem_audit": {
            "enabled": _siem_audit_enabled(),
            "log_path": str(_siem_audit_log_path()),
        },
        "prediction_log_path": str(_prediction_log_path()),
        "training_summary_path": str(_training_summary_path()),
        "drift_report_path": str(_drift_report_path()),
        "error": _load_error,
        "secondary_error": _secondary_load_error,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    base_url = _external_base_url(request)
    html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ArtPulse Live Endpoints</title>
    <style>
      :root {{
        --bg: #0b1320;
        --card: #111d33;
        --line: rgba(255, 255, 255, 0.14);
        --text: #edf2f7;
        --muted: #9fb3c8;
        --accent: #45a29e;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--text);
        font-family: "Inter", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at 12% 8%, rgba(69, 162, 158, 0.24), transparent 35%),
          radial-gradient(circle at 90% 0%, rgba(255, 183, 3, 0.18), transparent 30%),
          var(--bg);
      }}
      main {{
        max-width: 940px;
        margin: 0 auto;
        padding: 28px 18px 40px;
      }}
      h1 {{
        margin: 0;
        font-size: 30px;
      }}
      .sub {{
        margin-top: 8px;
        color: var(--muted);
      }}
      .grid {{
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
      }}
      .card {{
        background: linear-gradient(155deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px;
      }}
      .card h2 {{
        margin: 0 0 10px;
        font-size: 16px;
      }}
      ul {{
        margin: 0;
        padding-left: 18px;
      }}
      li {{
        margin: 6px 0;
      }}
      a {{
        color: #a8d8ff;
        text-decoration: none;
      }}
      a:hover {{ text-decoration: underline; }}
      pre {{
        margin: 8px 0 0;
        padding: 10px;
        border-radius: 10px;
        background: #0a111e;
        border: 1px solid var(--line);
        color: #d5e4f2;
        overflow-x: auto;
        font-size: 12px;
      }}
      .field {{
        margin-top: 8px;
      }}
      label {{
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 4px;
      }}
      input, button {{
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.05);
        color: var(--text);
        padding: 10px 12px;
        font-size: 13px;
      }}
      button {{
        margin-top: 8px;
        background: #45a29e;
        color: #061118;
        border: 0;
        font-weight: 700;
        cursor: pointer;
      }}
      .row {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }}
      .hint {{
        margin-top: 6px;
        color: var(--muted);
        font-size: 12px;
      }}
      @media (max-width: 820px) {{
        .grid {{
          grid-template-columns: 1fr;
        }}
        .row {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>ArtPulse Live Endpoints</h1>
      <div class="sub">Production ML inference API, demo token flow, and operations console.</div>

      <div class="grid">
        <section class="card">
          <h2>Service Status</h2>
          <ul>
            <li><a href="/health">GET /health</a></li>
            <li><a href="/ready">GET /ready</a></li>
            <li><a href="/metrics">GET /metrics</a></li>
          </ul>
        </section>
        <section class="card">
          <h2>Developer Access</h2>
          <ul>
            <li><a href="/docs">Swagger /docs</a></li>
            <li><a href="/openapi.json">OpenAPI /openapi.json</a></li>
            <li><a href="/admin">Ops Panel /admin</a></li>
          </ul>
        </section>
        <section class="card">
          <h2>Demo Flow</h2>
          <pre>1) POST /demo/token
2) GET /demo/status (Bearer token)
3) POST /demo/predict (Bearer token)</pre>
        </section>
        <section class="card">
          <h2>Try It (No CLI)</h2>
          <div class="field">
            <label>API Key (for /demo/token)</label>
            <input id="tryApiKey" type="password" placeholder="x-api-key" autocomplete="off" />
          </div>
          <div class="field">
            <label>Demo Subject</label>
            <input id="trySubject" type="text" value="portfolio-demo" />
          </div>
          <div class="row">
            <button id="btnToken" type="button">1) Mint Token</button>
            <button id="btnStatus" type="button">2) Demo Status</button>
          </div>
          <button id="btnPredict" type="button">3) Demo Predict</button>
          <div class="hint">This keeps token flow visible for client demos without terminal commands.</div>
          <pre id="tryOutput">Click "Mint Token" to start.</pre>
        </section>
        <section class="card">
          <h2>Base URL</h2>
          <pre>{base_url}</pre>
        </section>
      </div>
    </main>
    <script>
      let demoToken = "";
      const apiKeyEl = document.getElementById("tryApiKey");
      const subjectEl = document.getElementById("trySubject");
      const outEl = document.getElementById("tryOutput");

      function setOutput(payload) {{
        if (typeof payload === "string") {{
          outEl.textContent = payload;
          return;
        }}
        outEl.textContent = JSON.stringify(payload, null, 2);
      }}

      async function mintToken() {{
        const apiKey = apiKeyEl.value.trim();
        const subject = (subjectEl.value || "portfolio-demo").trim();
        if (!apiKey) {{
          setOutput("API key is required to mint token.");
          return;
        }}
        const res = await fetch("/demo/token", {{
          method: "POST",
          headers: {{
            "content-type": "application/json",
            "x-api-key": apiKey
          }},
          body: JSON.stringify({{ subject, ttl_seconds: 600 }})
        }});
        const payload = await res.json();
        if (!res.ok) {{
          setOutput(payload);
          return;
        }}
        demoToken = payload.access_token || "";
        setOutput({{
          step: "token_minted",
          expires_at: payload.expires_at,
          token_preview: demoToken ? `${{demoToken.slice(0, 16)}}...` : ""
        }});
      }}

      async function demoStatus() {{
        if (!demoToken) {{
          setOutput("Mint token first.");
          return;
        }}
        const res = await fetch("/demo/status", {{
          headers: {{ Authorization: `Bearer ${{demoToken}}` }}
        }});
        const payload = await res.json();
        setOutput(payload);
      }}

      async function demoPredict() {{
        if (!demoToken) {{
          setOutput("Mint token first.");
          return;
        }}
        const body = {{
          row: {{
            hue_mean: 0.62,
            sat_mean: 0.55,
            val_mean: 0.7,
            contrast: 0.4,
            edges: 0.12
          }}
        }};
        const res = await fetch("/demo/predict", {{
          method: "POST",
          headers: {{
            "content-type": "application/json",
            Authorization: `Bearer ${{demoToken}}`
          }},
          body: JSON.stringify(body)
        }});
        const payload = await res.json();
        setOutput(payload);
      }}

      document.getElementById("btnToken").addEventListener("click", () => {{
        mintToken().catch((err) => setOutput(`Token error: ${{err.message}}`));
      }});
      document.getElementById("btnStatus").addEventListener("click", () => {{
        demoStatus().catch((err) => setOutput(`Status error: ${{err.message}}`));
      }});
      document.getElementById("btnPredict").addEventListener("click", () => {{
        demoPredict().catch((err) => setOutput(`Predict error: ${{err.message}}`));
      }});
    </script>
  </body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/ready")
def ready() -> dict:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc
    return {
        "status": "ready",
        "mlflow_model_uri": _model_uri,
        "secondary_model_uri": _secondary_model_uri,
        "rollout_mode": _rollout_mode(),
    }


@app.post("/reload-model")
def reload_model(_: str = Depends(require_auth)) -> dict:
    try:
        load_model(force_reload=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
    return {
        "status": "reloaded",
        "mlflow_model_uri": _model_uri,
        "secondary_model_uri": _secondary_model_uri,
        "rollout_mode": _rollout_mode(),
    }


@app.get("/metadata")
def metadata(_: str = Depends(require_auth)) -> dict:
    return {
        "labels": LABELS,
        "feature_order": FEATURE_ORDER,
        "mlflow_model_uri": _model_uri,
        "secondary_model_uri": _secondary_model_uri,
        "rollout_mode": _rollout_mode(),
        "secondary_percent": _secondary_traffic_percent(),
    }


@app.get("/ops/summary")
def ops_summary(_: str = Depends(require_ops_access)) -> dict:
    return _build_ops_summary()


@app.get("/admin", response_class=HTMLResponse)
def admin_panel() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ArtPulse Ops Console</title>
    <style>
      :root {
        --bg: #0d1b2a;
        --bg-alt: #1b263b;
        --card: #14213d;
        --line: rgba(255, 255, 255, 0.12);
        --text: #f1faee;
        --muted: #b8c1d1;
        --accent: #ffb703;
        --good: #2a9d8f;
        --warn: #f4a261;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at 20% 15%, rgba(255, 183, 3, 0.16), transparent 40%),
          radial-gradient(circle at 80% 0%, rgba(42, 157, 143, 0.2), transparent 35%),
          linear-gradient(145deg, var(--bg) 0%, var(--bg-alt) 100%);
      }
      main {
        max-width: 1120px;
        margin: 0 auto;
        padding: 26px 22px 40px;
      }
      .head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 18px;
      }
      h1 {
        margin: 0;
        letter-spacing: 0.5px;
        font-size: 28px;
        font-weight: 700;
      }
      .sub {
        color: var(--muted);
        font-size: 13px;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(12, minmax(0, 1fr));
        gap: 14px;
      }
      .card {
        grid-column: span 4;
        background: linear-gradient(160deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px;
      }
      .card.wide { grid-column: span 6; }
      .card.full { grid-column: span 12; }
      .label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .value {
        margin-top: 6px;
        font-size: 24px;
        font-weight: 700;
        word-break: break-word;
      }
      .ok { color: var(--good); }
      .warn { color: var(--warn); }
      .mono {
        font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
        font-size: 12px;
        color: #d9e2ec;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
      }
      th, td {
        text-align: left;
        padding: 8px 6px;
        border-bottom: 1px solid var(--line);
        font-size: 13px;
      }
      th { color: var(--muted); font-weight: 600; }
      .meta {
        margin-top: 14px;
        color: var(--muted);
        font-size: 12px;
      }
      .auth-row {
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      .auth-row input {
        min-width: 320px;
        flex: 1 1 420px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.22);
        color: var(--text);
        padding: 10px 12px;
      }
      .auth-row button {
        border: 0;
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 600;
        cursor: pointer;
        background: var(--accent);
        color: #1d1d1d;
      }
      .auth-row button.ghost {
        border: 1px solid var(--line);
        background: transparent;
        color: var(--text);
      }
      @media (max-width: 900px) {
        .card, .card.wide, .card.full { grid-column: span 12; }
        .auth-row input {
          min-width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="head">
        <div>
          <h1>ArtPulse Ops Console</h1>
          <div class="sub">Model/version control, runtime KPI panel, drift trend, and retrain status</div>
        </div>
        <div id="ts" class="sub">Loading...</div>
      </div>
      <div class="grid">
        <section class="card full">
          <div class="label">Ops API Key</div>
          <div class="sub">Use `x-api-key` or corporate SSO (if OIDC header mode is enabled).</div>
          <div class="auth-row">
            <input id="apiKeyInput" type="password" placeholder="Paste API key" autocomplete="off" />
            <button id="saveKeyBtn" type="button">Save key</button>
            <button id="clearKeyBtn" type="button" class="ghost">Clear</button>
          </div>
          <div id="authHint" class="meta">Key is stored only in this browser.</div>
        </section>
        <section class="card">
          <div class="label">Primary Model</div>
          <div id="primaryModel" class="value mono">-</div>
        </section>
        <section class="card">
          <div class="label">Secondary Model</div>
          <div id="secondaryModel" class="value mono">-</div>
        </section>
        <section class="card">
          <div class="label">Rollout</div>
          <div id="rolloutMode" class="value">-</div>
        </section>
        <section class="card">
          <div class="label">Request Count</div>
          <div id="kpiRequests" class="value">0</div>
          <div id="kpiPredictions" class="meta">predictions: 0</div>
        </section>
        <section class="card">
          <div class="label">Error Rate</div>
          <div id="kpiErrorRate" class="value">0.00%</div>
          <div id="kpiErrors" class="meta">errors: 0</div>
        </section>
        <section class="card">
          <div class="label">P95 Latency</div>
          <div id="kpiP95" class="value">0 ms</div>
          <div id="kpiP95Inference" class="meta">inference p95: 0 ms</div>
        </section>
        <section class="card wide">
          <div class="label">Drift Score</div>
          <div id="driftScore" class="value">0.00</div>
          <div id="driftMeta" class="meta">No data</div>
        </section>
        <section class="card wide">
          <div class="label">Last Retrain</div>
          <div id="lastRetrain" class="value">-</div>
          <div id="retrainStatus" class="meta">status: unknown</div>
          <div id="rolloutPercent" class="meta">Secondary traffic: 0%</div>
        </section>
        <section class="card wide">
          <div class="label">Drift Trend</div>
          <div id="driftTrend" class="value">flat</div>
          <div id="driftTrendMeta" class="meta">delta: 0.000000</div>
        </section>
        <section class="card full">
          <div class="label">Registry Alias Versions</div>
          <table>
            <thead><tr><th>Alias</th><th>Version</th></tr></thead>
            <tbody id="aliasRows"></tbody>
          </table>
          <div class="meta">Tip: Use this page in client demos to show operational maturity.</div>
        </section>
      </div>
    </main>
    <script>
      let apiKey = localStorage.getItem("artpulse_api_key") || "";
      let refreshTimer = null;
      let oidcMode = false;
      const apiKeyInput = document.getElementById("apiKeyInput");
      const authHint = document.getElementById("authHint");
      const saveKeyBtn = document.getElementById("saveKeyBtn");
      const clearKeyBtn = document.getElementById("clearKeyBtn");

      if (apiKey) {
        apiKeyInput.value = apiKey;
        authHint.textContent = "API key loaded from browser storage.";
      }

      function setText(id, value) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = value ?? "-";
      }

      function setAuthHint(text, isError = false) {
        authHint.textContent = text;
        authHint.classList.remove("ok", "warn");
        authHint.classList.add(isError ? "warn" : "ok");
      }

      async function detectAuthMode() {
        try {
          const res = await fetch("/health");
          if (!res.ok) return;
          const payload = await res.json();
          oidcMode = Boolean(payload.auth?.ops_oidc_enabled);
          const allowedDomains = payload.auth?.ops_oidc_allowed_email_domains || [];
          if (oidcMode) {
            const domainHint = allowedDomains.length
              ? ` (allowed domains: ${allowedDomains.join(", ")})`
              : "";
            setAuthHint(`Corporate SSO mode is active${domainHint}. API key is optional fallback.`, false);
          }
        } catch (_) {
          // Ignore and keep API key mode fallback.
        }
      }

      function saveApiKey() {
        const value = apiKeyInput.value.trim();
        if (!value) {
          setAuthHint("API key is required.", true);
          return;
        }
        apiKey = value;
        localStorage.setItem("artpulse_api_key", value);
        setAuthHint("API key saved. Refreshing panel data...", false);
        refresh().catch((err) => {
          setText("ts", err.message);
          setAuthHint(err.message, true);
        });
      }

      function clearApiKey() {
        apiKey = "";
        localStorage.removeItem("artpulse_api_key");
        apiKeyInput.value = "";
        setAuthHint("API key cleared.", false);
        setText("ts", oidcMode ? "Using SSO mode..." : "Waiting for API key...");
      }

      function fmtDate(value) {
        if (!value || value === "unknown") return "unknown";
        const d = new Date(value);
        if (Number.isNaN(d.getTime())) return value;
        return d.toLocaleString();
      }

      async function refresh() {
        if (!apiKey && !oidcMode) {
          setText("ts", "Waiting for API key...");
          return;
        }

        const headers = {};
        if (apiKey) headers["x-api-key"] = apiKey;
        const res = await fetch("/ops/summary", { headers });
        if (res.status === 401) {
          if (oidcMode) {
            throw new Error("SSO session missing or expired. Re-login via corporate SSO.");
          }
          throw new Error("Invalid API key. Update key above.");
        }
        if (res.status === 403) throw new Error("SSO user not allowed by domain policy.");
        if (!res.ok) throw new Error("Failed to load /ops/summary");
        const payload = await res.json();

        setText("primaryModel", payload.model?.primary_uri || "not loaded");
        setText("secondaryModel", payload.model?.secondary_uri || "none");
        setText("rolloutMode", payload.rollout?.mode || "single");
        setText("lastRetrain", fmtDate(payload.model?.last_retrain_at));

        const kpi = payload.kpis || {};
        const reqCount = Number(kpi.request_count || 0);
        const predCount = Number(kpi.prediction_count || 0);
        const errRatePct = Number(kpi.error_rate_percent || 0);
        const errCount = Number(kpi.error_count || 0);
        const p95 = Number(kpi.p95_latency_ms || 0);
        const infP95 = Number(kpi.inference_p95_latency_ms || 0);
        setText("kpiRequests", reqCount.toLocaleString());
        setText("kpiPredictions", `predictions: ${predCount.toLocaleString()}`);
        setText("kpiErrorRate", `${errRatePct.toFixed(2)}%`);
        setText("kpiErrors", `errors: ${errCount.toLocaleString()} | auth_failures: ${Number(kpi.auth_failures || 0).toLocaleString()}`);
        setText("kpiP95", `${p95.toFixed(2)} ms`);
        setText("kpiP95Inference", `inference p95: ${infP95.toFixed(2)} ms`);

        const drift = payload.drift || {};
        const driftScore = Number(drift.drift_score || 0).toFixed(3);
        const driftEl = document.getElementById("driftScore");
        setText("driftScore", driftScore);
        driftEl.classList.remove("ok", "warn");
        driftEl.classList.add(drift.drift_detected ? "warn" : "ok");
        setText("driftMeta", `detected=${Boolean(drift.drift_detected)} | events=${drift.event_count || 0} | updated=${fmtDate(drift.generated_at)}`);

        const trend = drift.trend || {};
        const trendDirection = trend.direction || "flat";
        const trendDelta = Number(trend.delta || 0);
        const trendPoints = Array.isArray(trend.points) ? trend.points.length : 0;
        const driftTrendEl = document.getElementById("driftTrend");
        setText("driftTrend", trendDirection);
        setText("driftTrendMeta", `delta: ${trendDelta.toFixed(6)} | points: ${trendPoints}`);
        driftTrendEl.classList.remove("ok", "warn");
        driftTrendEl.classList.add(trendDirection === "up" ? "warn" : "ok");

        const retrain = payload.retrain || {};
        const dq = retrain.data_quality_passed;
        const dqText = dq === true ? "passed" : (dq === false ? "failed" : "n/a");
        setText(
          "retrainStatus",
          `status: ${retrain.status || "unknown"} | quality: ${dqText} | candidates: ${retrain.candidate_count ?? 0}`
        );
        setText("rolloutPercent", `Secondary traffic: ${payload.rollout?.secondary_percent ?? 0}%`);
        setText("ts", `Last refresh: ${new Date().toLocaleTimeString()}`);

        const aliasRows = document.getElementById("aliasRows");
        aliasRows.innerHTML = "";
        const aliases = payload.model?.alias_versions || {};
        Object.entries(aliases).forEach(([alias, version]) => {
          const row = document.createElement("tr");
          row.innerHTML = `<td>${alias}</td><td>${version}</td>`;
          aliasRows.appendChild(row);
        });

        setAuthHint(
          oidcMode ? "Connected: /ops/summary loaded via SSO/API key." : "Connected: /ops/summary loaded.",
          false
        );
      }

      async function boot() {
        saveKeyBtn.addEventListener("click", saveApiKey);
        clearKeyBtn.addEventListener("click", clearApiKey);
        apiKeyInput.addEventListener("keydown", (evt) => {
          if (evt.key === "Enter") {
            saveApiKey();
          }
        });

        await detectAuthMode();
        try {
          await refresh();
        } catch (err) {
          setText("ts", err.message);
          setAuthHint(err.message, true);
        }
        refreshTimer = setInterval(() => {
          refresh().catch((err) => {
            setText("ts", err.message);
            setAuthHint(err.message, true);
          });
        }, 15000);
      }

      boot();
    </script>
  </body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/demo/status")
def demo_status(identity: str = Depends(require_demo_access)) -> dict:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    return {
        "status": "ready",
        "demo_enabled": _demo_enabled(),
        "rollout_mode": _rollout_mode(),
        "secondary_percent": _secondary_traffic_percent(),
        "model_uri": _model_uri,
        "secondary_model_uri": _secondary_model_uri,
        "actor": identity,
    }


@app.post("/demo/token", response_model=DemoTokenResponse)
def demo_token(req: DemoTokenRequest, _: str = Depends(require_auth)) -> DemoTokenResponse:
    token, issued_at_epoch, expires_at_epoch, kid = _mint_demo_token(
        subject=req.subject,
        ttl_seconds=req.ttl_seconds,
    )
    issued_at_dt = datetime.fromtimestamp(issued_at_epoch, timezone.utc).isoformat()
    expires_at_dt = datetime.fromtimestamp(expires_at_epoch, timezone.utc).isoformat()
    return DemoTokenResponse(
        token_type="Bearer",
        access_token=token,
        expires_in=expires_at_epoch - issued_at_epoch,
        issued_at=issued_at_dt,
        expires_at=expires_at_dt,
        kid=kid,
        audience=_demo_token_audience(),
        issuer=_demo_token_issuer(),
    )


@app.post("/demo/predict", response_model=DemoPredictResponse)
def demo_predict(req: DemoPredictRequest, identity: str = Depends(require_demo_access)) -> DemoPredictResponse:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    row_values = [getattr(req.row, name) for name in FEATURE_ORDER]
    X_np = np.asarray([row_values], dtype=np.float32)
    routing_key = json.dumps(row_values, sort_keys=True)

    selected_model, selected_uri, route = _select_inference_model(
        identity=identity,
        routing_key=routing_key,
    )
    preds = selected_model.predict(X_np)
    pred_id = int(np.asarray(preds).tolist()[0])
    label = LABELS[pred_id] if 0 <= pred_id < len(LABELS) else "unknown"
    rollout_mode = _rollout_mode()

    feature_payload = {name: float(getattr(req.row, name)) for name in FEATURE_ORDER}
    _append_prediction_events(
        [feature_payload],
        [pred_id],
        [label],
        source="public_demo",
        actor=identity,
        model_uri=selected_uri,
        rollout_route=route,
        rollout_mode=rollout_mode,
    )
    _inc_prediction(source="public_demo", label=label)
    _inc_rollout_route(source="public_demo", mode=rollout_mode, route=route)

    demo_request_id = hashlib.sha256(f"{time.time_ns()}:{identity}".encode("utf-8")).hexdigest()[:16]
    return DemoPredictResponse(
        label=label,
        prediction=pred_id,
        mlflow_model_uri=selected_uri,
        rollout_route=route,
        rollout_mode=rollout_mode,
        demo_request_id=demo_request_id,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, identity: str = Depends(require_auth)) -> PredictResponse:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    rows = [[getattr(row, name) for name in FEATURE_ORDER] for row in req.rows]
    X_np = np.asarray(rows, dtype=np.float32)

    routing_key = json.dumps(rows, sort_keys=True)
    selected_model, selected_uri, route = _select_inference_model(
        identity=identity,
        routing_key=routing_key,
    )
    preds = selected_model.predict(X_np)

    preds_int = [int(p) for p in np.asarray(preds).tolist()]
    labels = [LABELS[i] if 0 <= i < len(LABELS) else "unknown" for i in preds_int]
    rollout_mode = _rollout_mode()

    row_dicts = [{name: float(getattr(row, name)) for name in FEATURE_ORDER} for row in req.rows]
    _append_prediction_events(
        row_dicts,
        preds_int,
        labels,
        source="tabular",
        actor=identity,
        model_uri=selected_uri,
        rollout_route=route,
        rollout_mode=rollout_mode,
    )

    for label in labels:
        _inc_prediction(source="tabular", label=label)
    _inc_rollout_route(source="tabular", mode=rollout_mode, route=route)

    return PredictResponse(
        labels=labels,
        predictions=preds_int,
        mlflow_model_uri=selected_uri,
        rollout_route=route,
        rollout_mode=rollout_mode,
    )


@app.post("/predict-image", response_model=PredictImageResponse)
def predict_image(req: PredictImageRequest, identity: str = Depends(require_auth)) -> PredictImageResponse:
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}") from exc

    try:
        content = base64.b64decode(req.image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        feature_vector = extract_image_features_from_bytes(
            image_bytes=content,
            image_size=_image_feature_size(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image parsing failed: {exc}") from exc

    X_np = np.asarray([feature_vector], dtype=np.float32)
    routing_key = hashlib.sha256(content[:4096]).hexdigest()
    selected_model, selected_uri, route = _select_inference_model(
        identity=identity,
        routing_key=routing_key,
    )
    preds = selected_model.predict(X_np)
    pred_id = int(np.asarray(preds).tolist()[0])
    label = LABELS[pred_id] if 0 <= pred_id < len(LABELS) else "unknown"
    rollout_mode = _rollout_mode()

    feature_payload = {
        FEATURE_ORDER[idx]: float(feature_vector[idx]) for idx in range(len(FEATURE_ORDER))
    }
    _append_prediction_events(
        [feature_payload],
        [pred_id],
        [label],
        source="image_upload",
        actor=identity,
        model_uri=selected_uri,
        rollout_route=route,
        rollout_mode=rollout_mode,
    )
    _inc_prediction(source="image_upload", label=label)
    _inc_rollout_route(source="image_upload", mode=rollout_mode, route=route)

    return PredictImageResponse(
        label=label,
        prediction=pred_id,
        features=feature_payload,
        mlflow_model_uri=selected_uri,
        rollout_route=route,
        rollout_mode=rollout_mode,
    )
