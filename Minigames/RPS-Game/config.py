import os
import shutil

WINDOW_TITLE  = "Rock Paper Scissors vs Computer"
WINDOW_WIDTH  = 960
WINDOW_HEIGHT = 540
FPS           = 30
ICON_SIZE     = 100   # px; R/P/S icon size blitted onto the pygame surface

# Camera backend — auto-detected: the Raspberry Pi ships the `rpicam-vid`
# binary (Pi Camera Module via pi_camera.RpiCamera); dev machines don't and
# fall back to a USB/built-in webcam. Override with env MINIGAME_CAMERA_BACKEND.
#   "opencv"  — USB webcam / built-in camera (macOS dev, Pi USB)
#   "rpicam"  — Raspberry Pi Camera Module via rpicam-vid (shared pi_camera.py)
#   "picamera2" — Pi Camera Module via picamera2 bindings (system Python only)
CAMERA_BACKEND = os.environ.get("MINIGAME_CAMERA_BACKEND") or (
    "rpicam" if shutil.which("rpicam-vid") else "opencv")
CAMERA_INDEX   = 0      # only used when CAMERA_BACKEND = "opencv"
FRAME_WIDTH    = 640    # used by rpicam / picamera2 backends
FRAME_HEIGHT   = 480

# Colors (R, G, B)
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GREEN  = ( 40, 210,  60)
RED    = (220,  50,  50)
YELLOW = (255, 215,   0)
GRAY   = (180, 180, 180)
