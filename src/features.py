import numpy as np

LABELS = ["minimal", "neo-pop", "surreal", "monochrome", "vibrant"]
FEATURE_ORDER = ["hue_mean", "sat_mean", "val_mean", "contrast", "edges"]

def make_synthetic_dataset(n: int = 2000, seed: int = 42):
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

    X = np.stack([hue_mean, sat_mean, val_mean, contrast, edges], axis=1)

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
