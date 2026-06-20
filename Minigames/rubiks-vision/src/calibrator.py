"""
calibrator.py — interactive calibration flow.

The user holds up each solved face in turn.  For each of the 6 standard
colours (white, yellow, red, orange, green, blue) the flow:
  1. Shows the live feed with the ROI guide and a prompt.
  2. Waits for SPACE to capture.
  3. Samples the centre sticker (index 4) of the warped face.
  4. Stores the median BGR and pre-computed LAB to calibration.json.

After all 6 colours are captured the file is written and the live loop ends.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import cv2
import numpy as np
from skimage.color import rgb2lab

if TYPE_CHECKING:
    from src.camera import CameraSource
    from src.face_detector import FaceDetector
    from src.sticker_sampler import StickerSampler


COLOR_NAMES = ["white", "yellow", "red", "orange", "green", "blue"]


def _bgr_to_lab(bgr: np.ndarray) -> list[float]:
    """Convert BGR uint8 (3,) → LAB float list for JSON storage."""
    rgb = bgr[::-1].astype(np.float32) / 255.0
    lab = rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)
    return [float(x) for x in lab]


def run_calibration(
    camera: "CameraSource",
    detector: "FaceDetector",
    sampler: "StickerSampler",
    calibration_path: str,
) -> None:
    """
    Drive the interactive calibration session.
    Blocks until all 6 colours are captured or the user presses ESC/q.
    """
    captured: dict[str, dict] = {}
    color_idx = 0
    total = len(COLOR_NAMES)

    print("\n=== Calibration Mode ===")
    print(
        "Present each face colour inside the guide box and press SPACE to capture."
    )
    print("Press ESC or q to abort.\n")

    while color_idx < total:
        current_color = COLOR_NAMES[color_idx]

        ok, frame = camera.read()
        if not ok:
            print("[calibrator] Camera read failed — retrying…")
            continue

        warped, corners, fallback = detector.detect(frame)
        samples = sampler.sample(warped)
        center = samples[4]  # index 4 = centre sticker

        # Draw overlay
        display = frame.copy()
        x, y, w, h = detector.guide_rect(frame)
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Swatch of current centre sample
        swatch_bgr = tuple(int(v) for v in center.bgr)
        cv2.rectangle(display, (10, 10), (60, 60), swatch_bgr, -1)
        cv2.rectangle(display, (10, 10), (60, 60), (255, 255, 255), 2)

        prompt = f"[{color_idx+1}/{total}] Show {current_color.upper()} face → SPACE to capture  |  q/ESC to quit"
        cv2.putText(
            display, prompt, (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
        )
        cv2.putText(
            display, prompt, (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        if fallback:
            cv2.putText(
                display, "Auto-detect failed — using guide box",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2
            )

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            print("[calibrator] Aborted.")
            cv2.destroyWindow("Calibration")
            return

        if key == ord(" "):
            bgr = center.bgr
            captured[current_color] = {
                "bgr": [int(bgr[0]), int(bgr[1]), int(bgr[2])],
                "lab": _bgr_to_lab(bgr),
            }
            print(f"  Captured {current_color}: BGR={bgr.tolist()}")
            color_idx += 1

    cv2.destroyWindow("Calibration")

    with open(calibration_path, "w") as f:
        json.dump(captured, f, indent=2)
    print(f"\nCalibration saved to {calibration_path!r}.")
    print("Re-run `python main.py run` to use the new palette.")
