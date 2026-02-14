from src.release_gate import evaluate_release_gate


def test_release_gate_pass(monkeypatch):
    monkeypatch.setattr("src.release_gate._prom_query", lambda _u, expr: 0.12 if "histogram_quantile" in expr else 0.005)
    monkeypatch.setattr(
        "src.release_gate._ops_summary",
        lambda _u, api_key: {"drift": {"drift_score": 1.2}, "rollout": {"mode": "canary"}},
    )

    report = evaluate_release_gate(
        api_url="https://api.staging.wearetheartmakers.com",
        api_key="k",
        prometheus_url="https://prom.example.com",
        p95_ms_limit=180.0,
        error_rate_limit=0.01,
        drift_score_limit=2.5,
    )
    assert report["passed"] is True


def test_release_gate_fail(monkeypatch):
    monkeypatch.setattr("src.release_gate._prom_query", lambda _u, expr: 0.30 if "histogram_quantile" in expr else 0.02)
    monkeypatch.setattr(
        "src.release_gate._ops_summary",
        lambda _u, api_key: {"drift": {"drift_score": 3.1}, "rollout": {"mode": "canary"}},
    )

    report = evaluate_release_gate(
        api_url="https://api.staging.wearetheartmakers.com",
        api_key="k",
        prometheus_url="https://prom.example.com",
        p95_ms_limit=180.0,
        error_rate_limit=0.01,
        drift_score_limit=2.5,
    )
    assert report["passed"] is False
    assert report["checks"]["drift_score"]["passed"] is False
