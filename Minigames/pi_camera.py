"""
pi_camera.py — shared Raspberry Pi Camera Module source for all minigames.

Why this exists
---------------
The games run in a Python 3.12 conda env (mediapipe needs <=3.12), but the
libcamera Python bindings (picamera2) only exist for the Pi's *system* Python.
Bridging that gap, this module reads raw YUV420 frames from the `rpicam-vid`
system binary over a pipe and decodes them to BGR with OpenCV. It works from
any Python because rpicam-vid is an external process.

Ironclad cleanup — the camera is NEVER left hanging
---------------------------------------------------
A leaked or hard-killed rpicam-vid wedges the IMX708 sensor and the next launch
black-screens. This module defends on both ends:

  * On startup it terminates any stale rpicam-vid (SIGTERM, not SIGKILL) and
    waits for the kernel to release the sensor.
  * The child runs in its OWN process group, and release() group-terminates it
    gracefully (SIGTERM, wait, then SIGKILL only as a last resort) so libcamera
    can hand the sensor back cleanly.
  * release() is also wired to atexit, so it runs on normal return, sys.exit,
    an unhandled exception, or a propagated KeyboardInterrupt (Ctrl-C).

Usage
-----
    from pi_camera import RpiCamera
    cam = RpiCamera(640, 480)          # flip=False (default)
    ok, bgr = cam.read()               # non-blocking; (False, None) until ready
    cam.release()
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time

import cv2
import numpy as np


# rpicam-vid needs a moment after a previous instance dies before the sensor
# is free again; and a moment after launch before it streams the first frame.
_SENSOR_RELEASE_SEC = 0.3


def _kill_stale_rpicam() -> None:
    """Gracefully terminate any leftover rpicam-vid so the sensor is free."""
    subprocess.run(["pkill", "-TERM", "-f", "rpicam-vid"],
                   stderr=subprocess.DEVNULL)
    time.sleep(_SENSOR_RELEASE_SEC)


def _terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM the child's whole process group, escalating to SIGKILL only if
    it refuses to exit. Graceful first so libcamera releases the sensor."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class RpiCamera:
    """rpicam-vid frame source with a cv2.VideoCapture-compatible interface.

    read()    -> (ok, bgr_frame)   non-blocking; (False, None) until first frame
    release() -> None              idempotent; always frees the camera
    isOpened()/ __enter__/__exit__ provided for drop-in compatibility.
    """

    def __init__(self, width: int, height: int, *,
                 framerate: int = 30, flip: bool = False) -> None:
        self._w = width
        self._h = height
        self._flip = flip
        self._frame_bytes = width * height * 3 // 2  # I420 planar (YUV420)

        _kill_stale_rpicam()

        self._proc = subprocess.Popen(
            [
                "rpicam-vid", "-t", "0",
                "--width", str(width), "--height", str(height),
                "--codec", "yuv420", "--framerate", str(framerate),
                "-n", "-o", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # own process group → clean group-kill later
        )

        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._released = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        # Belt-and-suspenders: free the camera even if the game forgets to.
        atexit.register(self.release)

    def _reader(self) -> None:
        stream = self._proc.stdout
        n = self._frame_bytes
        while self._running:
            # read(n) on the buffered pipe blocks until exactly n bytes arrive,
            # so a partial pipe-buffer read never yields a torn frame.
            raw = stream.read(n)
            if not raw or len(raw) < n:
                break  # EOF — rpicam-vid exited
            yuv = np.frombuffer(raw, np.uint8).reshape((self._h * 3 // 2, self._w))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            if self._flip:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def isOpened(self) -> bool:
        return self._proc.poll() is None

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._running = False
        _terminate_group(self._proc)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # Context-manager support (rubiks-vision uses `with create_camera(...) as cam`)
    def __enter__(self) -> "RpiCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
