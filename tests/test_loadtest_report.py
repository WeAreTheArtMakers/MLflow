from src.loadtest_report import build_markdown_report


def test_build_markdown_report_contains_kpis():
    summary = {
        "metrics": {
            "http_req_duration": {"values": {"avg": 54.2, "p(95)": 102.8, "p(99)": 140.1}},
            "http_reqs": {"values": {"rate": 38.5}},
            "http_req_failed": {"values": {"rate": 0.01}},
            "checks": {"values": {"rate": 1.0}},
        }
    }
    report = build_markdown_report(summary)
    assert "Load Test Report" in report
    assert "P95 latency" in report
    assert "38.50" in report
