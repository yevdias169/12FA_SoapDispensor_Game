"""
color_classifier.py — classify a BGR colour against the calibrated palette.

Primary method (CLASSIFIER="lab"):
  Convert BGR → RGB → CIE-LAB, compare each palette entry with
  skimage.color.deltaE_ciede2000. Pick the entry with smallest ΔE.
  If smallest ΔE > DELTA_E_MAX → label "unknown".

Fallback method (CLASSIFIER="hsv"):
  Check each calibrated HSV band (H_center ± H_half, S_min–S_max,
  V_min–V_max).  Less robust to lighting variation; provided for
  comparison and as an offline sanity check.

Why CIE-LAB + CIEDE2000?
  CIEDE2000 is perceptually uniform — a ΔE of 1 corresponds roughly
  to a just-noticeable difference regardless of hue, saturation, or
  lightness region.  RGB/HSV distances are not perceptually uniform and
  produce poor discrimination near white/grey or at low saturation.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert a single BGR uint8 colour (shape (3,)) to CIE-LAB."""
    rgb = bgr[::-1].astype(np.float32) / 255.0          # BGR→RGB, normalise
    rgb_img = rgb.reshape(1, 1, 3)                        # skimage expects HxWx3
    lab = rgb2lab(rgb_img)                                # returns float64 HxWx3
    return lab.reshape(3).astype(np.float64)


def _bgr_to_hsv_norm(bgr: np.ndarray) -> tuple[float, float, float]:
    """Return (H [0,180], S [0,255], V [0,255]) for a BGR pixel."""
    pixel = bgr.reshape(1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).reshape(3)
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


# ---------------------------------------------------------------------------
# Palette representation
# ---------------------------------------------------------------------------

class Palette:
    """
    Holds the reference colours used for classification.
    Each entry stores the colour name, its BGR (uint8), and its pre-computed
    LAB value, so we avoid re-computing LAB on every frame.
    """

    def __init__(self, entries: dict[str, tuple[int, int, int]]) -> None:
        """
        entries: mapping colour_name → (B, G, R) uint8.
        """
        self.names: list[str] = []
        self.lab_values: list[np.ndarray] = []
        self.bgr_values: list[np.ndarray] = []
        for name, bgr_tuple in entries.items():
            bgr = np.array(bgr_tuple, dtype=np.uint8)
            self.names.append(name)
            self.bgr_values.append(bgr)
            self.lab_values.append(_bgr_to_lab(bgr))

    def __len__(self) -> int:
        return len(self.names)


def load_palette(calibration_path: str, default_palette: dict[str, Any]) -> Palette:
    """
    Load palette from calibration.json if it exists, else use default_palette.
    calibration.json stores entries as { "name": {"bgr": [B,G,R]} }.
    """
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            data = json.load(f)
        entries: dict[str, tuple[int, int, int]] = {}
        for name, vals in data.items():
            b, g, r = vals["bgr"]
            entries[name] = (int(b), int(g), int(r))
        return Palette(entries)

    warnings.warn(
        f"calibration.json not found at {calibration_path!r}. "
        "Using default reference colours — accuracy may be poor. "
        "Run `python main.py calibrate` to calibrate for your cube and lighting.",
        stacklevel=2,
    )
    return Palette(default_palette)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class LabClassifier:
    """CIEDE2000-based classifier (primary)."""

    def __init__(self, palette: Palette, delta_e_max: float) -> None:
        self.palette = palette
        self.delta_e_max = delta_e_max

    def classify(self, bgr: np.ndarray) -> tuple[str, float]:
        """
        Return (label, best_delta_e).
        label is "unknown" if best_delta_e > delta_e_max.
        """
        lab = _bgr_to_lab(bgr)
        best_label = "unknown"
        best_de = float("inf")
        for name, ref_lab in zip(self.palette.names, self.palette.lab_values):
            de = float(
                deltaE_ciede2000(
                    lab.reshape(1, 1, 3), ref_lab.reshape(1, 1, 3)
                ).item()
            )
            if de < best_de:
                best_de = de
                best_label = name
        if best_de > self.delta_e_max:
            return "unknown", best_de
        return best_label, best_de

    def delta_e_between(self, bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
        """Compute CIEDE2000 ΔE between two BGR colours."""
        lab_a = _bgr_to_lab(bgr_a).reshape(1, 1, 3)
        lab_b = _bgr_to_lab(bgr_b).reshape(1, 1, 3)
        return float(deltaE_ciede2000(lab_a, lab_b).item())

    def max_pairwise_delta_e(self, bgr_list: list[np.ndarray]) -> float:
        """Return the maximum pairwise ΔE among a list of BGR colours."""
        if len(bgr_list) < 2:
            return 0.0
        labs = [_bgr_to_lab(c).reshape(1, 1, 3) for c in bgr_list]
        max_de = 0.0
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                de = float(deltaE_ciede2000(labs[i], labs[j]).item())
                if de > max_de:
                    max_de = de
        return max_de


class HsvClassifier:
    """
    HSV range-based classifier (fallback).
    Each palette entry specifies (H_center, H_half, S_min, S_max, V_min, V_max).
    Red wraps around H=0/180 so it is checked with two windows.
    """

    def __init__(
        self,
        hsv_palette: dict[str, tuple[int, int, int, int, int, int]],
    ) -> None:
        self.hsv_palette = hsv_palette

    def classify(self, bgr: np.ndarray) -> tuple[str, float]:
        """Return (label, 0.0). 0.0 is a placeholder distance (HSV has no ΔE)."""
        h, s, v = _bgr_to_hsv_norm(bgr)
        for name, (hc, hh, s_min, s_max, v_min, v_max) in self.hsv_palette.items():
            if not (s_min <= s <= s_max and v_min <= v <= v_max):
                continue
            # Check H in [hc-hh, hc+hh] mod 180
            lo = (hc - hh) % 180
            hi = (hc + hh) % 180
            if lo <= hi:
                in_range = lo <= h <= hi
            else:  # wraps around 0/180
                in_range = h >= lo or h <= hi
            if in_range:
                return name, 0.0
        return "unknown", 0.0

    def delta_e_between(self, bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
        """HSV fallback has no perceptual ΔE; returns 0 to satisfy interface."""
        return 0.0

    def max_pairwise_delta_e(self, bgr_list: list[np.ndarray]) -> float:
        return 0.0


def create_classifier(
    mode: str,
    palette: Palette,
    delta_e_max: float,
    hsv_palette: dict[str, tuple[int, int, int, int, int, int]],
) -> LabClassifier | HsvClassifier:
    """Factory: return the right classifier based on config.CLASSIFIER."""
    if mode == "lab":
        return LabClassifier(palette, delta_e_max)
    elif mode == "hsv":
        return HsvClassifier(hsv_palette)
    else:
        raise ValueError(f"Unknown CLASSIFIER={mode!r}. Choose 'lab' or 'hsv'.")
