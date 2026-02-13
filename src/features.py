from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

LABELS = ["minimal", "neo-pop", "surreal", "monochrome", "vibrant"]
FEATURE_ORDER = ["hue_mean", "sat_mean", "val_mean", "contrast", "edges"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_synthetic_dataset(n: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic "art style" feature dataset.
    Features (0..1):
      - hue_mean
      - sat_mean
      - val_mean
      - contrast
      - edges (texture/complexity proxy)
    """
    rng = np.random.default_rng(seed)

    hue_mean = rng.uniform(0, 1, n)
    sat_mean = rng.uniform(0, 1, n)
    val_mean = rng.uniform(0, 1, n)
    contrast = rng.uniform(0, 1, n)
    edges = rng.uniform(0, 1, n)

    X = np.stack([hue_mean, sat_mean, val_mean, contrast, edges], axis=1).astype(np.float32)

    y = np.zeros(n, dtype=int)

    mono = (sat_mean < 0.25) & (val_mean > 0.35) & (val_mean < 0.85)
    y[mono] = 3

    minimal = (edges < 0.25) & (contrast < 0.30) & (~mono)
    y[minimal] = 0

    vibrant = (sat_mean > 0.70) & (val_mean > 0.60)
    y[vibrant] = 4

    neopop = (sat_mean > 0.60) & (contrast > 0.65) & (~vibrant)
    y[neopop] = 1

    surreal = (edges > 0.55) & ((hue_mean < 0.15) | (hue_mean > 0.85))
    y[surreal] = 2

    noise_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    y[noise_idx] = rng.integers(0, len(LABELS), size=len(noise_idx))

    return X, y


def _iter_image_paths(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def extract_image_features_from_pil(image: Image.Image, image_size: int = 128) -> np.ndarray:
    """Extract compact handcrafted features from a PIL image.

    The returned values are normalized to approximately 0..1.
    """
    rgb = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    hsv = rgb.convert("HSV")

    hsv_np = np.asarray(hsv, dtype=np.float32) / 255.0
    gray_np = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

    hue_mean = float(hsv_np[..., 0].mean())
    sat_mean = float(hsv_np[..., 1].mean())
    val_mean = float(hsv_np[..., 2].mean())

    # Std is bounded near 0.5 for grayscale values in [0,1].
    contrast = float(np.clip(gray_np.std() * 2.0, 0.0, 1.0))

    gx = np.abs(np.diff(gray_np, axis=1)).mean() if gray_np.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray_np, axis=0)).mean() if gray_np.shape[0] > 1 else 0.0
    edges = float(np.clip((gx + gy) / 2.0, 0.0, 1.0))

    return np.asarray([hue_mean, sat_mean, val_mean, contrast, edges], dtype=np.float32)


def extract_image_features_from_path(image_path: str | Path, image_size: int = 128) -> np.ndarray:
    with Image.open(image_path) as image:
        return extract_image_features_from_pil(image=image, image_size=image_size)


def extract_image_features_from_bytes(image_bytes: bytes, image_size: int = 128) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        return extract_image_features_from_pil(image=image, image_size=image_size)


def build_image_feature_dataset(
    dataset_dir: str,
    labels: Sequence[str] = LABELS,
    image_size: int = 128,
    min_images_per_label: int = 5,
    strict_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Build a tabular feature dataset from real images.

    Expected directory structure:
      dataset_dir/
        minimal/*.jpg
        neo-pop/*.jpg
        surreal/*.jpg
        monochrome/*.jpg
        vibrant/*.jpg
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    class_counts: Dict[str, int] = {}

    for label in labels:
        if label not in LABEL_TO_INDEX:
            continue

        class_dir = root / label
        if not class_dir.exists():
            if strict_labels:
                raise ValueError(f"Missing class directory: {class_dir}")
            continue

        image_paths = list(_iter_image_paths(class_dir))
        if not image_paths:
            if strict_labels:
                raise ValueError(f"No images in class directory: {class_dir}")
            continue

        if len(image_paths) < min_images_per_label:
            raise ValueError(
                f"Class '{label}' has {len(image_paths)} images; "
                f"minimum required is {min_images_per_label}."
            )

        class_counts[label] = len(image_paths)
        label_index = LABEL_TO_INDEX[label]

        for image_path in image_paths:
            X_rows.append(extract_image_features_from_path(image_path=image_path, image_size=image_size))
            y_rows.append(label_index)

    if not X_rows:
        raise ValueError("No labeled images found. Check dataset_dir and class folder names.")

    unique_labels = set(y_rows)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes for training.")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=int)

    metadata: Dict[str, object] = {
        "dataset_dir": str(root.resolve()),
        "image_size": image_size,
        "total_images": int(len(y_rows)),
        "class_counts": class_counts,
    }
    return X, y, metadata
