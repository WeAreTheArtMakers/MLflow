import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.features import LABELS


def _rand_color(rng: np.random.Generator, low: int = 0, high: int = 255) -> tuple[int, int, int]:
    return (int(rng.integers(low, high)), int(rng.integers(low, high)), int(rng.integers(low, high)))


def _generate_minimal(rng: np.random.Generator, size: int) -> Image.Image:
    bg = int(rng.integers(190, 245))
    img = Image.new("RGB", (size, size), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    for _ in range(2):
        x1 = int(rng.integers(0, size // 2))
        y1 = int(rng.integers(0, size // 2))
        x2 = int(rng.integers(size // 2, size))
        y2 = int(rng.integers(size // 2, size))
        c = int(np.clip(bg + rng.integers(-20, 20), 0, 255))
        draw.rectangle([x1, y1, x2, y2], outline=(c, c, c), width=2)
    return img


def _generate_neo_pop(rng: np.random.Generator, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), _rand_color(rng, 120, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(8):
        x1 = int(rng.integers(0, size - 32))
        y1 = int(rng.integers(0, size - 32))
        x2 = int(np.clip(x1 + rng.integers(32, size // 2), 0, size))
        y2 = int(np.clip(y1 + rng.integers(32, size // 2), 0, size))
        draw.rectangle([x1, y1, x2, y2], fill=_rand_color(rng, 120, 255), outline=(0, 0, 0), width=2)
    return img


def _generate_surreal(rng: np.random.Generator, size: int) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    arr[..., 0] = np.clip(255 * np.abs(np.sin(4 * np.pi * xv)), 0, 255)
    arr[..., 1] = np.clip(255 * np.abs(np.cos(3 * np.pi * yv)), 0, 255)
    arr[..., 2] = np.clip(255 * np.abs(np.sin(2 * np.pi * (xv + yv))), 0, 255)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        radius = int(rng.integers(size // 12, size // 5))
        cx = int(rng.integers(radius, size - radius))
        cy = int(rng.integers(radius, size - radius))
        color = _rand_color(rng, 20, 255)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=color, width=4)
    return img.filter(ImageFilter.DETAIL)


def _generate_monochrome(rng: np.random.Generator, size: int) -> Image.Image:
    base = int(rng.integers(80, 200))
    img = Image.new("L", (size, size), base)
    draw = ImageDraw.Draw(img)
    for _ in range(7):
        c = int(np.clip(base + rng.integers(-60, 60), 0, 255))
        x1 = int(rng.integers(0, size - 20))
        y1 = int(rng.integers(0, size - 20))
        x2 = int(np.clip(x1 + rng.integers(20, size // 2), 0, size))
        y2 = int(np.clip(y1 + rng.integers(20, size // 2), 0, size))
        draw.ellipse([x1, y1, x2, y2], fill=c)
    return img.convert("RGB")


def _generate_vibrant(rng: np.random.Generator, size: int) -> Image.Image:
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    arr[..., 1] = np.clip(arr[..., 1] + 60, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] + 50, 0, 255)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(5):
        y = int(rng.integers(0, size))
        draw.line([(0, y), (size, y)], fill=_rand_color(rng, 140, 255), width=3)
    return img


def generate_dataset(output_dir: str, images_per_label: int, image_size: int, seed: int) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    generators = {
        "minimal": _generate_minimal,
        "neo-pop": _generate_neo_pop,
        "surreal": _generate_surreal,
        "monochrome": _generate_monochrome,
        "vibrant": _generate_vibrant,
    }

    for label in LABELS:
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(images_per_label):
            rng = np.random.default_rng(seed + (idx * 37) + (LABELS.index(label) * 10000))
            image = generators[label](rng, image_size)
            image.save(label_dir / f"{label}_{idx:04d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a real-image style dataset for ArtPulse")
    parser.add_argument("--output-dir", default="data/images")
    parser.add_argument("--images-per-label", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        images_per_label=args.images_per_label,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Dataset generated at: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
