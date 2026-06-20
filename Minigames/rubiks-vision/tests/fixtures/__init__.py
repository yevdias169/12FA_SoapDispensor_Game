"""
fixtures/__init__.py — synthetic fixture generator.

Call make_fixtures() at the top of each test module to ensure PNG files exist.
All images are generated in-memory using numpy and written here; CI needs no
binary assets committed to the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

FIXTURE_DIR = Path(__file__).parent


def _uniform_face(bgr: tuple[int, int, int], size: int = 300) -> np.ndarray:
    """Return a size×size BGR image painted uniformly with bgr."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def _mixed_face(
    colors: list[tuple[int, int, int]],
    size: int = 300,
) -> np.ndarray:
    """
    Return a size×size BGR image with a 3×3 grid of colours.
    colors must have exactly 9 entries (row-major).
    """
    assert len(colors) == 9
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // 3
    for i, bgr in enumerate(colors):
        row, col = divmod(i, 3)
        y, x = row * cell, col * cell
        h = cell if row < 2 else size - y
        w = cell if col < 2 else size - x
        img[y : y + h, x : x + w] = bgr
    return img


def make_fixtures() -> dict[str, Path]:
    """
    Generate fixture PNGs if they don't already exist.
    Returns a mapping name → absolute path.
    """
    files: dict[str, Path] = {}

    specs: list[tuple[str, np.ndarray]] = [
        ("uniform_white.png",  _uniform_face((255, 255, 255))),
        ("uniform_blue.png",   _uniform_face((200, 50,  10 ))),
        ("uniform_red.png",    _uniform_face((0,   0,   200))),
        # Mixed: 8 white stickers + 1 blue centre
        ("mixed_one_off.png",  _mixed_face(
            [(255, 255, 255)] * 4 + [(200, 50, 10)] + [(255, 255, 255)] * 4
        )),
        # Fully mixed: every cell a different colour
        ("mixed_full.png",     _mixed_face([
            (255, 255, 255), (0, 255, 255), (0, 0, 200),
            (0, 128, 255),   (0, 200, 0),   (200, 0, 0),
            (255, 255, 255), (0, 255, 255), (0, 0, 200),
        ])),
    ]

    for filename, img in specs:
        p = FIXTURE_DIR / filename
        if not p.exists():
            cv2.imwrite(str(p), img)
        files[filename.replace(".png", "")] = p

    return files
