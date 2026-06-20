import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')

# Paths reference the original files in place — do not copy or move them
MODEL_PATH          = os.path.join(_ROOT, 'model.p')
HAND_LANDMARKER_PATH = os.path.join(_ROOT, 'hand_landmarker.task')

# Camera backend:
#   "opencv"    — USB webcam / built-in camera (macOS, Pi USB)
#   "picamera2" — Raspberry Pi Camera Module via ribbon cable
CAMERA_BACKEND = "picamera2"

# Resolution used when capturing frames.
# picamera2 requires explicit dimensions; opencv uses these as hints.
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Only used when CAMERA_BACKEND = "opencv".
# Existing scripts (test_classifier.py, Data_Collection.py) use index 1.
CAMERA_INDEX = 1

WINDOW_TITLE  = "Finger Guessing Game"
WINDOW_WIDTH  = 960
WINDOW_HEIGHT = 540
FPS           = 30

# Colors (R, G, B)
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GREEN  = ( 40, 210,  60)
RED    = (220,  50,  50)
YELLOW = (255, 215,   0)
GRAY   = (180, 180, 180)
