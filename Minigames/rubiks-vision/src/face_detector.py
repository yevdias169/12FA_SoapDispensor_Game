"""
face_detector.py — locate the Rubik's face in a frame and warp it to a square.

Two modes (config.DETECTION_MODE):
  "guided"  — user aligns cube inside a fixed on-screen ROI rectangle.
  "auto"    — detect the face quad via Canny + contour heuristics; falls back
              to guided ROI when no valid quad is found.

In both modes the output is a WARP_SIZE × WARP_SIZE BGR image where the
face exactly fills the frame — everything downstream is geometry-agnostic.
"""

from __future__ import annotations

import numpy as np
import cv2


def _guided_roi(frame: np.ndarray, roi_frac: float) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the centred square guide ROI."""
    h, w = frame.shape[:2]
    side = int(min(w, h) * roi_frac)
    x = (w - side) // 2
    y = (h - side) // 2
    return x, y, side, side


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 points, return them in order: top-left, top-right, bottom-right,
    bottom-left.  Robust to arbitrary input ordering.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    # Sort by sum (top-left has smallest sum, bottom-right has largest)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()  # y - x; top-right has smallest
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def _warp_to_square(frame: np.ndarray, corners: np.ndarray, size: int) -> np.ndarray:
    """Perspective-warp the quad defined by corners to a size×size image."""
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, M, (size, size))


def _auto_detect_quad(
    frame: np.ndarray, min_area_frac: float
) -> np.ndarray | None:
    """
    Try to find the face quad via Canny edges + contour filtering.
    Returns ordered 4×2 float32 corners, or None if no valid quad found.
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Median-based Canny thresholds — adapts to overall image brightness
    median_val = float(np.median(blurred))
    lower = max(0, int(0.66 * median_val))
    upper = int(1.33 * median_val)
    edges = cv2.Canny(blurred, lower, upper)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_area = 0.0
    best_corners: np.ndarray | None = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_frac * frame_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best_corners = _order_corners(approx)

    return best_corners


class FaceDetector:
    """Detects/crops the Rubik's face and warps it to a canonical square."""

    def __init__(
        self,
        mode: str,
        roi_frac: float,
        warp_size: int,
        min_face_area_frac: float,
    ) -> None:
        self.mode = mode
        self.roi_frac = roi_frac
        self.warp_size = warp_size
        self.min_face_area_frac = min_face_area_frac

    def detect(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Detect the face in frame.

        Returns:
            warped        — WARP_SIZE × WARP_SIZE BGR face image.
            corners       — 4×2 float32 corners in original frame coords.
            used_fallback — True when auto mode fell back to the guided ROI.
        """
        used_fallback = False

        if self.mode == "auto":
            corners = _auto_detect_quad(frame, self.min_face_area_frac)
            if corners is None:
                used_fallback = True

        if self.mode == "guided" or used_fallback:
            x, y, w, h = _guided_roi(frame, self.roi_frac)
            corners = _order_corners(
                np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            )

        warped = _warp_to_square(frame, corners, self.warp_size)
        return warped, corners, used_fallback

    def guide_rect(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """Return the guided ROI rect for drawing purposes."""
        return _guided_roi(frame, self.roi_frac)
