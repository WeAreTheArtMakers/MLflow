import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class QualityConfig:
    min_samples: int = 50
    min_per_label: int = 5
    min_classes: int = 2
    feature_min: float = 0.0
    feature_max: float = 1.0
    max_imbalance_ratio: float = 5.0


def _append_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def _class_distribution(y: np.ndarray, labels: Sequence[str]) -> Dict[str, int]:
    counts: Counter[int] = Counter(int(item) for item in y.tolist())
    dist: Dict[str, int] = {}
    for idx, label in enumerate(labels):
        dist[label] = int(counts.get(idx, 0))
    return dist


def run_quality_checks(
    X: Any,
    y: Any,
    feature_order: Sequence[str],
    labels: Sequence[str],
    dataset_meta: Dict[str, Any],
    config: QualityConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or QualityConfig()

    X_np = np.asarray(X)
    y_np = np.asarray(y).astype(int)
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    shape_ok = X_np.ndim == 2 and X_np.shape[1] == len(feature_order)
    _append_check(
        checks,
        "feature_schema",
        shape_ok,
        f"expected_features={len(feature_order)}, observed_shape={tuple(X_np.shape)}",
    )
    if not shape_ok:
        errors.append("Feature schema mismatch.")

    sample_count_ok = int(X_np.shape[0]) >= cfg.min_samples
    _append_check(
        checks,
        "min_samples",
        sample_count_ok,
        f"min_samples={cfg.min_samples}, observed={int(X_np.shape[0])}",
    )
    if not sample_count_ok:
        errors.append("Insufficient sample size.")

    label_alignment_ok = X_np.shape[0] == y_np.shape[0]
    _append_check(
        checks,
        "label_alignment",
        label_alignment_ok,
        f"X_rows={int(X_np.shape[0])}, y_rows={int(y_np.shape[0])}",
    )
    if not label_alignment_ok:
        errors.append("X and y row counts do not match.")

    finite_ok = bool(np.isfinite(X_np).all())
    _append_check(checks, "finite_values", finite_ok, "all_features_finite=true required")
    if not finite_ok:
        errors.append("Dataset contains NaN or infinite values.")

    in_range_mask = (X_np >= cfg.feature_min) & (X_np <= cfg.feature_max)
    in_range_ok = bool(in_range_mask.all())
    out_of_range_count = int((~in_range_mask).sum())
    _append_check(
        checks,
        "feature_range",
        in_range_ok,
        f"expected=[{cfg.feature_min}, {cfg.feature_max}], out_of_range={out_of_range_count}",
    )
    if not in_range_ok:
        errors.append("Feature values outside expected normalized range.")

    class_counts = _class_distribution(y_np, labels)
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    class_count_ok = len(non_zero_counts) >= cfg.min_classes
    _append_check(
        checks,
        "min_class_count",
        class_count_ok,
        f"min_classes={cfg.min_classes}, observed={len(non_zero_counts)}",
    )
    if not class_count_ok:
        errors.append("Insufficient number of represented classes.")

    min_per_label_ok = all(count >= cfg.min_per_label for count in non_zero_counts)
    _append_check(
        checks,
        "min_samples_per_label",
        min_per_label_ok,
        f"min_per_label={cfg.min_per_label}, observed={class_counts}",
    )
    if not min_per_label_ok:
        errors.append("One or more classes are under the minimum sample threshold.")

    imbalance_ratio = 1.0
    if non_zero_counts:
        imbalance_ratio = float(max(non_zero_counts) / max(1, min(non_zero_counts)))
    imbalance_ok = imbalance_ratio <= cfg.max_imbalance_ratio
    _append_check(
        checks,
        "class_imbalance",
        imbalance_ok,
        f"max_ratio={cfg.max_imbalance_ratio}, observed={imbalance_ratio:.3f}",
    )
    if not imbalance_ok:
        warnings.append(
            f"Class imbalance ratio is high ({imbalance_ratio:.3f}). Consider balancing strategy."
        )

    image_meta = dataset_meta.get("image_meta", {}) if isinstance(dataset_meta, dict) else {}
    image_counts = image_meta.get("class_counts", {}) if isinstance(image_meta, dict) else {}
    if image_counts:
        missing_classes = [label for label in labels if int(image_counts.get(label, 0)) == 0]
        image_labels_ok = not missing_classes
        _append_check(
            checks,
            "image_label_coverage",
            image_labels_ok,
            f"missing_labels={missing_classes}",
        )
        if not image_labels_ok:
            errors.append("Image dataset is missing required class folders.")

    passed = len(errors) == 0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "summary": {
            "n_samples": int(X_np.shape[0]),
            "n_features": int(X_np.shape[1]) if X_np.ndim == 2 else 0,
            "class_counts": class_counts,
            "imbalance_ratio": float(imbalance_ratio),
        },
    }


def save_quality_report(report: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
