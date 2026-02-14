import numpy as np

from src.data_quality import QualityConfig, run_quality_checks
from src.features import FEATURE_ORDER, LABELS, make_synthetic_dataset


def test_quality_checks_pass_for_synthetic_dataset():
    X, y = make_synthetic_dataset(n=1200, seed=7)
    report = run_quality_checks(
        X=X,
        y=y,
        feature_order=FEATURE_ORDER,
        labels=LABELS,
        dataset_meta={"dataset_type": "synthetic"},
        config=QualityConfig(min_samples=500, min_per_label=5),
    )
    assert report["passed"] is True
    assert report["summary"]["n_samples"] == 1200


def test_quality_checks_fail_out_of_range_features():
    X, y = make_synthetic_dataset(n=600, seed=11)
    X = np.copy(X)
    X[0, 0] = 9.0
    report = run_quality_checks(
        X=X,
        y=y,
        feature_order=FEATURE_ORDER,
        labels=LABELS,
        dataset_meta={"dataset_type": "synthetic"},
        config=QualityConfig(min_samples=500, min_per_label=5),
    )
    assert report["passed"] is False
    assert any("normalized range" in msg for msg in report["errors"])
