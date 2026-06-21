"""
config.py — single source of truth for all tunable parameters.
CAMERA_BACKEND is auto-detected (Pi camera vs webcam); override with the
environment variable MINIGAME_CAMERA_BACKEND=opencv|rpicam|picamera2.
"""

import os
import shutil

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# Auto-detect: the Raspberry Pi ships the `rpicam-vid` binary (Pi Camera Module
# via the shared pi_camera.RpiCamera); dev machines fall back to a webcam.
CAMERA_BACKEND: str = os.environ.get("MINIGAME_CAMERA_BACKEND") or (
    "rpicam" if shutil.which("rpicam-vid") else "opencv")
CAMERA_INDEX: int = -1           # -1 = auto-detect first working camera (handles macOS Continuity Camera)
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
DETECTION_MODE: str = "guided"   # "guided" (robust default) | "auto" (contour-based)

# guided mode: ROI square edge = ROI_FRAC * min(frame_width, frame_height)
ROI_FRAC: float = 0.6

# auto mode: reject contours smaller than this fraction of the frame area
MIN_FACE_AREA_FRAC: float = 0.1

# Warped face size in pixels (square); all sticker work happens here
WARP_SIZE: int = 300

# ---------------------------------------------------------------------------
# Sticker sampling
# ---------------------------------------------------------------------------
# Central fraction of each 3×3 cell used for colour sampling (avoids bevels)
PATCH_FRAC: float = 0.5

# Glare detection: a patch is flagged if mean V > this and mean S < this
GLARE_V_THRESH: float = 0.92
GLARE_S_THRESH: float = 0.08

# ---------------------------------------------------------------------------
# Colour classification
# ---------------------------------------------------------------------------
CLASSIFIER: str = "lab"          # "lab" (CIEDE2000) | "hsv" (range-based fallback)

# lab: max ΔE to accept a palette match; above this → "unknown"
# Tune up if solved faces are misclassified; tune down to be stricter
DELTA_E_MAX: float = 20.0

# Max ΔE spread among the 9 stickers for the face to count as "uniform"
# Tune up if a valid face is rejected; tune down for stricter uniformity
INTRA_FACE_DELTA_E: float = 12.0

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
STABLE_FRAMES: int = 8           # consecutive agreeing frames before emitting verdict
MAX_UNCERTAIN: int = 1           # glare-flagged stickers above this → RETRY verdict

# ---------------------------------------------------------------------------
# Default palette (BGR, used when calibration.json is absent)
# These are broad reference values; calibration always wins.
# ---------------------------------------------------------------------------
DEFAULT_PALETTE: dict[str, tuple[int, int, int]] = {
    "white":  (255, 255, 255),
    "yellow": (0,   255, 255),
    "red":    (0,   0,   200),
    "orange": (0,   128, 255),
    "green":  (0,   200, 0  ),
    "blue":   (200, 0,   0  ),
}

# HSV tolerance bands used by the fallback classifier
# Each entry: (H_center, H_half_range, S_min, S_max, V_min, V_max)
# H is in OpenCV [0,180] convention
HSV_PALETTE: dict[str, tuple[int, int, int, int, int, int]] = {
    "white":  (0,   180, 0,   40,  200, 255),
    "yellow": (30,  10,  100, 255, 100, 255),
    "red":    (0,   10,  100, 255, 100, 255),   # also wrap-around ~170
    "orange": (10,  8,   100, 255, 100, 255),
    "green":  (60,  20,  50,  255, 50,  255),
    "blue":   (110, 20,  50,  255, 50,  255),
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
CALIBRATION_PATH: str = "calibration.json"

# ---------------------------------------------------------------------------
# Pygame minigame (rubiks_checker.py)
# ---------------------------------------------------------------------------
FPS: int = 60
WINDOW_SIZE: tuple[int, int] = (1280, 720)

# Frames the StableTracker must see the same result before auto-verifying a face
# Tune down for faster response; tune up if false positives occur
STABLE_CLEAR_FRAMES: int = 6    # consecutive "cleared" frames before next face allowed

# Toast animation timing (seconds)
TOAST_IN: float = 0.25          # fade+slide-in duration
TOAST_HOLD: float = 1.10        # hold duration at full opacity
TOAST_OUT: float = 0.45         # fade-out duration

# Final "CUBE SOLVE VERIFIED" overlay timing (seconds)
FINAL_BACKDROP_FADE: float = 0.40   # dark backdrop fade-in
FINAL_POP_TIME: float = 0.35        # text scale-pop duration
FINAL_AUTO_DISMISS: float = 3.0     # auto-close after this many seconds

# The six expected face colours (any order); cube is complete when all are captured
EXPECTED_COLORS: list[str] = ["white", "yellow", "red", "orange", "green", "blue"]

# When True the process exits cleanly (camera released) once the "CUBE SOLVED"
# overlay finishes.  Flip to False when integrating into a master game loop so
# the scene hands control back via finished=True instead of killing the process.
TERMINATE_ON_COMPLETE: bool = True
