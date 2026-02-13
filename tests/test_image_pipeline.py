from pathlib import Path

from src.features import FEATURE_ORDER, LABELS, build_image_feature_dataset
from src.generate_image_dataset import generate_dataset
from src.train import train_models


def test_build_image_feature_dataset(tmp_path):
    dataset_dir = tmp_path / "images"
    generate_dataset(
        output_dir=str(dataset_dir),
        images_per_label=8,
        image_size=96,
        seed=123,
    )

    X, y, meta = build_image_feature_dataset(
        dataset_dir=str(dataset_dir),
        image_size=64,
        min_images_per_label=5,
    )

    assert X.shape[1] == len(FEATURE_ORDER)
    assert len(X) == len(y)
    assert set(meta["class_counts"].keys()) == set(LABELS)


def test_train_models_with_image_dataset(tmp_path):
    dataset_dir = tmp_path / "images"
    output_dir = tmp_path / "artifacts"
    tracking_dir = tmp_path / "mlruns"

    generate_dataset(
        output_dir=str(dataset_dir),
        images_per_label=10,
        image_size=96,
        seed=42,
    )

    summary = train_models(
        tracking_uri=f"file:{tracking_dir}",
        experiment_name="artpulse_image_test",
        dataset_type="image",
        dataset_dir=str(dataset_dir),
        image_size=64,
        min_images_per_label=5,
        n_samples=200,
        seed=42,
        test_size=0.2,
        output_dir=str(output_dir),
        register_best=False,
    )

    assert summary["dataset"]["dataset_type"] == "image"
    assert summary["best_model_uri"].startswith("runs:/")
    assert (Path(output_dir) / "latest_model_uri.txt").exists()
    assert (Path(output_dir) / "baseline_feature_stats.json").exists()
