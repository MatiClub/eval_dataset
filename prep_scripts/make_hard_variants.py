#!/usr/bin/env python3
"""Create degraded "hard" variants of existing corpus images.

Simulates real-world capture conditions in a document management system:
skewed scans, motion blur, dim phone photos, sensor noise, and low-resolution
uploads. Variants are written next to their source with a HARD-<op>- prefix so
they stay in the same category (relevance labels are unchanged) while stressing
model robustness. Deterministic: fixed sources, fixed parameters.

Requires: pillow (already a project dependency).

Usage:
    python3 prep_scripts/make_hard_variants.py [--data-dir data]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def op_rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(245, 245, 245))


def op_blur(img: Image.Image, radius: float = 2.6) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def op_dark(img: Image.Image) -> Image.Image:
    """Dim, warm-tinted photo taken in bad indoor lighting."""
    dark = ImageEnhance.Brightness(img).enhance(0.45)
    dark = ImageEnhance.Color(dark).enhance(0.85)
    overlay = Image.new("RGB", dark.size, (60, 45, 10))
    return Image.blend(dark, overlay, 0.12)


def op_noise(img: Image.Image, sigma: float = 22.0, seed: int = 7) -> Image.Image:
    arr = np.asarray(img).astype(np.int16)
    rng = np.random.default_rng(seed)
    noisy = arr + rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def op_lowres(img: Image.Image, factor: int = 5) -> Image.Image:
    small = img.resize((max(1, img.width // factor), max(1, img.height // factor)),
                       Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR)


# (relative source path, op name, op fn)
VARIANTS = [
    ("invoice-photo/Invoice_1.jpg", "rotate", lambda im: op_rotate(im, 7.0)),
    ("invoice-photo/Invoice_22.jpg", "blur", op_blur),
    ("receipt-photo/1000-receipt.jpg", "dark", op_dark),
    ("receipt-photo/1005-receipt.jpg", "noise", op_noise),
    ("receipt-photo/1010-receipt.jpg", "rotate", lambda im: op_rotate(im, -5.5)),
    ("identity-photo/est_id_22.jpg", "blur", op_blur),
    ("identity-photo/svk_id_20.jpg", "dark", op_dark),
    ("diploma-photo/79-58441.jpg", "rotate", lambda im: op_rotate(im, -6.0)),
    ("medical-photo/1_5.jpg", "lowres", op_lowres),
    ("food-photo/PMD-food_photo-0001.jpg", "lowres", op_lowres),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="dataset root directory")
    args = parser.parse_args()

    root = Path(args.data_dir)
    for rel, op_name, op in VARIANTS:
        src = root / rel
        if not src.exists():
            print(f"skip (missing source): {src}")
            continue
        img = Image.open(src).convert("RGB")
        dst = src.with_name(f"HARD-{op_name}-{src.stem}.jpg")
        op(img).save(dst, quality=85)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
