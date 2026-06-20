"""
test_color_classifier.py — offline tests for colour classification.

No camera or physical cube needed.
"""

from __future__ import annotations

import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import config
from src.color_classifier import LabClassifier, HsvClassifier, Palette, _bgr_to_lab


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_palette() -> Palette:
    return Palette(config.DEFAULT_PALETTE)


@pytest.fixture()
def lab_classifier(default_palette) -> LabClassifier:
    return LabClassifier(default_palette, delta_e_max=config.DELTA_E_MAX)


@pytest.fixture()
def hsv_classifier() -> HsvClassifier:
    return HsvClassifier(config.HSV_PALETTE)


# ---------------------------------------------------------------------------
# LAB classifier
# ---------------------------------------------------------------------------

class TestLabClassifier:
    def test_white_classifies_as_white(self, lab_classifier):
        bgr = np.array([255, 255, 255], dtype=np.uint8)
        label, de = lab_classifier.classify(bgr)
        assert label == "white", f"Got {label!r} (ΔE={de:.2f})"

    def test_pure_blue_classifies_as_blue(self, lab_classifier):
        # BGR = (200, 0, 0) → blue reference
        bgr = np.array([200, 0, 0], dtype=np.uint8)
        label, de = lab_classifier.classify(bgr)
        assert label == "blue", f"Got {label!r} (ΔE={de:.2f})"

    def test_pure_red_classifies_as_red(self, lab_classifier):
        bgr = np.array([0, 0, 200], dtype=np.uint8)
        label, de = lab_classifier.classify(bgr)
        assert label == "red", f"Got {label!r} (ΔE={de:.2f})"

    def test_pure_green_classifies_as_green(self, lab_classifier):
        bgr = np.array([0, 200, 0], dtype=np.uint8)
        label, de = lab_classifier.classify(bgr)
        assert label == "green", f"Got {label!r} (ΔE={de:.2f})"

    def test_extreme_delta_e_yields_unknown(self, lab_classifier):
        # Pure magenta is not in the palette; expect "unknown" at tight threshold
        strict = LabClassifier(
            lab_classifier.palette, delta_e_max=5.0  # very tight
        )
        bgr = np.array([200, 0, 200], dtype=np.uint8)  # magenta BGR
        label, de = strict.classify(bgr)
        assert label == "unknown", (
            f"Expected 'unknown' with tight threshold, got {label!r} (ΔE={de:.2f})"
        )

    def test_delta_e_between_identical_is_zero(self, lab_classifier):
        bgr = np.array([128, 64, 32], dtype=np.uint8)
        de = lab_classifier.delta_e_between(bgr, bgr)
        assert de < 0.01, f"ΔE of identical colours should be ~0, got {de}"

    def test_delta_e_between_white_and_black(self, lab_classifier):
        white = np.array([255, 255, 255], dtype=np.uint8)
        black = np.array([0, 0, 0], dtype=np.uint8)
        de = lab_classifier.delta_e_between(white, black)
        # Should be a large value (>50 in CIEDE2000 scale)
        assert de > 40, f"Expected large ΔE for white vs black, got {de}"

    def test_max_pairwise_with_single_colour(self, lab_classifier):
        bgr = np.array([100, 100, 100], dtype=np.uint8)
        assert lab_classifier.max_pairwise_delta_e([bgr]) == 0.0

    def test_max_pairwise_with_uniform_list(self, lab_classifier):
        bgr = np.array([255, 0, 0], dtype=np.uint8)
        colours = [bgr.copy() for _ in range(9)]
        de = lab_classifier.max_pairwise_delta_e(colours)
        assert de < 1.0, f"Uniform list should have near-zero spread, got {de}"

    def test_max_pairwise_with_mixed_list(self, lab_classifier):
        white = np.array([255, 255, 255], dtype=np.uint8)
        black = np.array([0, 0, 0], dtype=np.uint8)
        de = lab_classifier.max_pairwise_delta_e([white, black])
        assert de > 40, f"White vs black spread should be large, got {de}"


# ---------------------------------------------------------------------------
# HSV classifier
# ---------------------------------------------------------------------------

class TestHsvClassifier:
    def test_white_classifies_as_white(self, hsv_classifier):
        bgr = np.array([255, 255, 255], dtype=np.uint8)
        label, _ = hsv_classifier.classify(bgr)
        assert label == "white", f"Got {label!r}"

    def test_saturated_blue_classifies_as_blue(self, hsv_classifier):
        # Strong blue: H≈110, S≈255, V≈255 in OpenCV HSV
        bgr = np.array([220, 30, 0], dtype=np.uint8)
        label, _ = hsv_classifier.classify(bgr)
        assert label == "blue", f"Got {label!r}"

    def test_unknown_for_unseen_colour(self, hsv_classifier):
        # Pure magenta won't match any HSV band
        bgr = np.array([180, 0, 180], dtype=np.uint8)
        label, _ = hsv_classifier.classify(bgr)
        assert label == "unknown", f"Got {label!r}"


# ---------------------------------------------------------------------------
# bgr_to_lab utility
# ---------------------------------------------------------------------------

class TestBgrToLab:
    def test_white_has_high_L(self):
        lab = _bgr_to_lab(np.array([255, 255, 255], dtype=np.uint8))
        assert lab[0] > 95, f"White L* should be ~100, got {lab[0]}"

    def test_black_has_low_L(self):
        lab = _bgr_to_lab(np.array([0, 0, 0], dtype=np.uint8))
        assert lab[0] < 5, f"Black L* should be ~0, got {lab[0]}"

    def test_shape(self):
        lab = _bgr_to_lab(np.array([128, 64, 32], dtype=np.uint8))
        assert lab.shape == (3,)
