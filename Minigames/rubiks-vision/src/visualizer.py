"""
visualizer.py — draw all overlays on the live frame.

Draws:
  • The ROI/face quad guide.
  • 9 sticker cell boxes with colour swatches and labels.
  • HUD: verdict, stability progress bar, mode, keyboard hints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.sticker_sampler import StickerSample
    from src.verifier import FaceResult, Verdict


# Colour map for verdicts (BGR)
_VERDICT_COLORS: dict[str, tuple[int, int, int]] = {
    "SOLVED":    (0, 220, 0),
    "UNSOLVED":  (0, 50, 220),
    "RETRY":     (0, 165, 255),
    "ANALYZING": (180, 180, 180),
}

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "white":   (255, 255, 255),
    "yellow":  (0,   230, 255),
    "red":     (0,   0,   220),
    "orange":  (0,   140, 255),
    "green":   (0,   200, 0  ),
    "blue":    (220, 0,   0  ),
    "unknown": (60,  60,  60 ),
}


def _text_with_outline(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float,
    fg: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw text with a dark outline for readability on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness)


def draw_guide(
    frame: np.ndarray,
    roi_rect: tuple[int, int, int, int],
    corners: np.ndarray | None,
    used_fallback: bool,
) -> None:
    """Draw ROI guide rectangle (and detected quad if auto mode found one)."""
    x, y, w, h = roi_rect
    # Always draw the guide box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if corners is not None and not used_fallback:
        # Draw detected quad in green
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    if used_fallback:
        _text_with_outline(
            frame, "Auto-detect fallback to guide", (x, y - 8),
            0.45, (0, 140, 255)
        )


def draw_stickers(
    frame: np.ndarray,
    samples: "list[StickerSample]",
    labels: list[str],
    warped_offset: tuple[int, int],
    scale_x: float,
    scale_y: float,
    offending_indices: list[int],
) -> None:
    """
    Draw the 9 sticker boxes on the frame.

    warped_offset, scale_x, scale_y map from warped-face coords back to
    original frame coordinates.
    """
    for sample, label in zip(samples, labels):
        cx, cy, cw, ch = sample.cell_rect
        # Map warped coords → frame coords
        fx = int(warped_offset[0] + cx * scale_x)
        fy = int(warped_offset[1] + cy * scale_y)
        fw = int(cw * scale_x)
        fh = int(ch * scale_y)

        is_bad = sample.index in offending_indices
        border_color = (0, 0, 220) if is_bad else (200, 200, 200)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), border_color, 2)

        # Small colour swatch in top-left corner of the cell
        swatch_size = max(8, int(min(fw, fh) * 0.35))
        swatch_bgr = tuple(int(v) for v in sample.bgr)
        cv2.rectangle(
            frame,
            (fx + 2, fy + 2),
            (fx + 2 + swatch_size, fy + 2 + swatch_size),
            swatch_bgr, -1,
        )
        cv2.rectangle(
            frame,
            (fx + 2, fy + 2),
            (fx + 2 + swatch_size, fy + 2 + swatch_size),
            (0, 0, 0), 1,
        )

        # Label text
        lbl_color = _LABEL_COLORS.get(label, (200, 200, 200))
        _text_with_outline(
            frame, label[:3],  # abbreviated to fit in the cell
            (fx + 4, fy + fh - 6),
            0.35, lbl_color,
        )

        # Glare warning
        if sample.uncertain:
            _text_with_outline(frame, "?!", (fx + fw - 18, fy + 14), 0.4, (0, 165, 255))


def draw_hud(
    frame: np.ndarray,
    result: "FaceResult",
    stability: int,
    stable_frames: int,
    mode: str,
) -> None:
    """Draw the verdict HUD at the top of the frame."""
    from src.verifier import Verdict

    verdict_str = result.verdict.name
    color = _VERDICT_COLORS.get(verdict_str, (200, 200, 200))

    # Background banner
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)

    # Verdict text
    _text_with_outline(frame, verdict_str, (10, 32), 1.0, color, 2)

    # Stability bar
    bar_x = 220
    bar_w = 180
    bar_h = 16
    bar_y = 14
    filled = int(bar_w * min(stability, stable_frames) / stable_frames)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
    _text_with_outline(
        frame, f"{stability}/{stable_frames}",
        (bar_x + bar_w + 6, bar_y + 13), 0.4, (220, 220, 220)
    )

    # Reasoning (offending stickers)
    if result.verdict == Verdict.UNSOLVED and result.offending_indices:
        reason = "Mismatch @ stickers: " + ", ".join(
            f"#{i}({c})" for i, c in zip(result.offending_indices, result.offending_colors)
        )
        _text_with_outline(frame, reason, (10, h - 30), 0.42, (80, 160, 255))

    if result.verdict == Verdict.RETRY:
        _text_with_outline(
            frame, f"Glare on {result.uncertain_count} sticker(s) — re-present face",
            (10, h - 30), 0.42, (0, 165, 255)
        )

    # Mode + controls
    controls = "SPACE: capture  |  c: calibrate  |  q/ESC: quit"
    _text_with_outline(frame, f"Mode: {mode}  |  {controls}", (10, h - 10), 0.38, (180, 180, 180))


def compute_warp_mapping(
    corners: np.ndarray,
    warp_size: int,
) -> tuple[tuple[int, int], float, float]:
    """
    Compute the offset and scale factors to map warped-face coordinates
    back to the original frame for drawing sticker boxes.

    Returns (offset_xy, scale_x, scale_y).
    """
    tl = corners[0]
    tr = corners[1]
    bl = corners[3]
    width_px = float(np.linalg.norm(tr - tl))
    height_px = float(np.linalg.norm(bl - tl))
    scale_x = width_px / warp_size
    scale_y = height_px / warp_size
    offset = (int(tl[0]), int(tl[1]))
    return offset, scale_x, scale_y
