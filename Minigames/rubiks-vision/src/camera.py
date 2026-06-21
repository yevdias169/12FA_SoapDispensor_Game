"""
camera.py — CameraSource abstraction for macOS (OpenCV) and Raspberry Pi (picamera2).

Switching from macOS to Pi ribbon-cable camera:
  1. sudo apt install -y python3-picamera2
  2. In config.py, set CAMERA_BACKEND = "picamera2"
  USB webcams on the Pi stay on CAMERA_BACKEND = "opencv".
"""

from __future__ import annotations

import abc
import os
import sys
from types import TracebackType
from typing import Type

import cv2
import numpy as np

# Shared Pi-camera module lives in the Minigames/ directory (three levels up:
# Minigames/rubiks-vision/src/camera.py -> Minigames/).
_MINIGAMES_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _MINIGAMES_DIR not in sys.path:
    sys.path.insert(0, _MINIGAMES_DIR)


class CameraSource(abc.ABC):
    """Abstract camera: delivers BGR frames."""

    @abc.abstractmethod
    def read(self) -> tuple[bool, np.ndarray]:
        """Return (ok, bgr_frame). If ok is False the frame is garbage."""

    @abc.abstractmethod
    def release(self) -> None:
        """Free underlying resources."""

    # Context-manager interface so callers can use `with create_camera(cfg) as cam:`
    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class OpenCVCamera(CameraSource):
    """Wraps cv2.VideoCapture. Works on macOS and for USB webcams on the Pi."""

    def __init__(self, index: int, width: int, height: int) -> None:
        # If index is -1 (auto), probe 0..3 and use the first that opens.
        # This handles macOS Continuity Camera, which shifts the built-in
        # webcam's index unpredictably depending on whether an iPhone is nearby.
        if index == -1:
            for probe in range(4):
                cap = cv2.VideoCapture(probe)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        self._cap = cap
                        print(f"[camera] Auto-selected camera index {probe}")
                        break
                    cap.release()
            else:
                raise RuntimeError(
                    "No working camera found on indices 0-3. "
                    "Check that a webcam is connected and not in use."
                )
        else:
            self._cap = cv2.VideoCapture(index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera at index {index}. "
                    "Check that a webcam is connected and not in use."
                )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[bool, np.ndarray]:
        ok, frame = self._cap.read()
        if ok:
            frame = cv2.flip(frame, 1)  # horizontal mirror — feels natural on webcam
        return ok, frame

    def release(self) -> None:
        self._cap.release()


class PiCamera2Camera(CameraSource):
    """
    Wraps picamera2 for the Pi Camera Module (ribbon cable).
    picamera2 is imported lazily so this module is safe to import on macOS.
    """

    def __init__(self, width: int, height: int) -> None:
        try:
            from picamera2 import Picamera2  # lazy import — Pi only
        except ImportError as exc:
            raise ImportError(
                "picamera2 is not installed. On Raspberry Pi: "
                "sudo apt install -y python3-picamera2"
            ) from exc

        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"format": "XRGB8888", "size": (width, height)}
        )
        self._cam.configure(cfg)
        self._cam.start()

    def read(self) -> tuple[bool, np.ndarray]:
        # picamera2 returns an XRGB array (4 channels, channel 0 is padding)
        frame = self._cam.capture_array()
        # Convert XRGB → BGR: drop padding channel, reverse RGB→BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return True, cv2.flip(bgr, 1)  # horizontal mirror — feels natural on webcam

    def release(self) -> None:
        self._cam.stop()


def create_camera(cfg) -> CameraSource:  # type: ignore[type-arg]
    """Factory: return the right CameraSource based on cfg.CAMERA_BACKEND."""
    backend = cfg.CAMERA_BACKEND
    if backend == "opencv":
        return OpenCVCamera(cfg.CAMERA_INDEX, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
    elif backend == "picamera2":
        return PiCamera2Camera(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
    elif backend == "rpicam":
        # Shared ironclad Pi Camera Module source. flip=True so read() returns
        # mirrored frames, matching OpenCVCamera/PiCamera2Camera behaviour.
        from pi_camera import RpiCamera
        return RpiCamera(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT, flip=True)
    else:
        raise ValueError(
            f"Unknown CAMERA_BACKEND={backend!r}. "
            "Choose 'opencv', 'rpicam', or 'picamera2'."
        )
