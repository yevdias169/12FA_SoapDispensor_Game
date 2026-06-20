"""
sticker_sampler.py — extract the 9 sticker dominant colours from a warped face.

The warped face (WARP_SIZE × WARP_SIZE) is divided into a 3×3 grid.
Only the central PATCH_FRAC portion of each cell is sampled to avoid
bevel shadows, inter-sticker gaps, and edge glare.
Dominant colour is the per-channel median (outlier-robust vs mean).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class StickerSample:
    """Result for a single sticker."""
    index: int                          # 0–8, row-major (0=top-left, 8=bottom-right)
    bgr: np.ndarray                     # shape (3,), dtype uint8 — dominant colour
    cell_rect: tuple[int, int, int, int]  # (x, y, w, h) in warped-face coords
    patch_rect: tuple[int, int, int, int]  # sampled sub-rect in warped-face coords
    uncertain: bool                     # True if glare detected


def _central_patch(
    x: int, y: int, w: int, h: int, frac: float
) -> tuple[int, int, int, int]:
    """Shrink a rect to its central frac×frac portion."""
    pw = max(1, int(w * frac))
    ph = max(1, int(h * frac))
    px = x + (w - pw) // 2
    py = y + (h - ph) // 2
    return px, py, pw, ph


def _is_glare(
    patch_bgr: np.ndarray,
    v_thresh: float,
    s_thresh: float,
) -> bool:
    """
    Return True if the patch looks like specular glare.
    Glare = very bright AND almost achromatic (desaturated).
    Thresholds are in [0,1] because we normalise before checking.
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_norm = hsv[:, :, 2] / 255.0
    s_norm = hsv[:, :, 1] / 255.0
    mean_v = float(np.mean(v_norm))
    mean_s = float(np.mean(s_norm))
    return mean_v > v_thresh and mean_s < s_thresh


class StickerSampler:
    """Samples 9 sticker colours from a warped face image."""

    def __init__(
        self,
        warp_size: int,
        patch_frac: float,
        glare_v_thresh: float,
        glare_s_thresh: float,
    ) -> None:
        self.warp_size = warp_size
        self.patch_frac = patch_frac
        self.glare_v_thresh = glare_v_thresh
        self.glare_s_thresh = glare_s_thresh

    def sample(self, warped: np.ndarray) -> list[StickerSample]:
        """
        Extract 9 StickerSamples from the warped face.

        warped: BGR image of shape (WARP_SIZE, WARP_SIZE, 3).
        Returns a list of 9 StickerSamples in row-major order.
        """
        size = self.warp_size
        cell_w = size // 3
        cell_h = size // 3
        results: list[StickerSample] = []

        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                cx = col * cell_w
                cy = row * cell_h
                # Last cell absorbs rounding remainder
                cw = cell_w if col < 2 else size - cx
                ch = cell_h if row < 2 else size - cy

                px, py, pw, ph = _central_patch(cx, cy, cw, ch, self.patch_frac)
                patch = warped[py : py + ph, px : px + pw]

                # Per-channel median — robust to glare pixels and shadows at edges
                dominant = np.median(
                    patch.reshape(-1, 3), axis=0
                ).astype(np.uint8)

                uncertain = _is_glare(patch, self.glare_v_thresh, self.glare_s_thresh)

                results.append(
                    StickerSample(
                        index=idx,
                        bgr=dominant,
                        cell_rect=(cx, cy, cw, ch),
                        patch_rect=(px, py, pw, ph),
                        uncertain=uncertain,
                    )
                )

        return results
