"""
frame_grabber.py — threaded camera reader.

Wraps a CameraSource in a daemon thread that continuously grabs frames and
stores the latest one.  The pygame scene pulls the latest frame each tick via
read_latest(), keeping the UI at a smooth framerate regardless of camera FPS.

This also eliminates the stale-frame problem that contributed to the re-trigger
bug: when the main thread was doing heavy work the ring-buffer debouncer was
being fed the same old frame repeatedly, making it look like the verdict held.
"""

from __future__ import annotations

import threading
import time

import numpy as np


class FrameGrabber:
    """Background thread that keeps the latest camera frame fresh."""

    def __init__(self, camera) -> None:  # camera: CameraSource
        self._camera = camera
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._camera.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            # Cap the grab loop at ~200 Hz. For blocking cameras (OpenCV) this
            # is a no-op; for already-threaded sources (RpiCamera) whose read()
            # returns instantly, it prevents a 100%-CPU busy-spin on the Pi.
            time.sleep(0.005)

    def read_latest(self) -> np.ndarray | None:
        """Return the most recent BGR frame, or None if none captured yet."""
        with self._lock:
            return self._frame

    def stop(self) -> None:
        """Stop the capture thread and release the underlying camera."""
        self._running = False
        self._thread.join(timeout=2.0)
        self._camera.release()
