"""
Cat Tracing Game
- Camera:  Raspberry Pi Camera Module 3 via picamera2 (falls back to cv2 on non-Pi)
- Tracker: MediaPipe Face Mesh — nose tip (landmark 4)
- Draw:    Open mouth = pen down, close mouth = pen up
- Score:   Chamfer distance vs the reference cat image

Controls:
  Open mouth      — draw (nose traces the line)
  Close mouth     — lift pen (reposition)
  S               — compute & display score
  C               — clear canvas
  ESC / Q         — quit

Setup:
  1. Place cat.png (the reference cat image) in the same folder as this script.
  2. On Raspberry Pi:   sudo apt install -y python3-picamera2
                        pip install pygame opencv-python mediapipe scipy numpy
  3. On desktop/other: pip install pygame opencv-python mediapipe scipy numpy
                        (picamera2 import fails gracefully → uses webcam instead)
"""

import os
import shutil
import sys
from pathlib import Path
import numpy as np
import cv2
import dlib
import pygame
from scipy.ndimage import distance_transform_edt

# Shared Pi-camera module lives in the Minigames/ directory (one level up).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MINIGAMES_DIR = os.path.dirname(_HERE)
if _MINIGAMES_DIR not in sys.path:
    sys.path.insert(0, _MINIGAMES_DIR)

# Camera backend — auto-detected: the Raspberry Pi ships the `rpicam-vid`
# binary (Pi Camera Module via pi_camera.RpiCamera); dev machines fall back to
# a webcam. Override with env MINIGAME_CAMERA_BACKEND=opencv|rpicam.
CAMERA_BACKEND = os.environ.get("MINIGAME_CAMERA_BACKEND") or (
    "rpicam" if shutil.which("rpicam-vid") else "opencv")

# ── constants ────────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 960, 640
CAM_W, CAM_H   = 640, 480
DRAW_COLOR     = (255, 70,  70)   # red strokes
CURSOR_OPEN    = (80,  230,  80)  # green dot  = mouth open  = drawing
CURSOR_CLOSED  = (180, 180, 180)  # grey  dot  = mouth closed = hovering
CAT_TINT       = (100, 160, 255)  # blue reference overlay
OVERLAY_ALPHA  = 65               # 0-255; lower = more ghost-like
BG_COLOR       = (22,  22,  22)
LINE_THICK     = 5

# Mouth-open threshold (normalised by inter-eye distance for scale invariance)
MOUTH_OPEN_RATIO = 0.18   # tune up/down if detection is too sensitive

CAT_IMAGE      = Path(__file__).parent / "cat2.png"


# ── camera helpers ───────────────────────────────────────────────────────────
def open_camera():
    """Open the camera matching CAMERA_BACKEND. Both the Pi-camera source and
    cv2.VideoCapture expose .read() -> (ok, bgr) and .release()."""
    if CAMERA_BACKEND == "rpicam":
        from pi_camera import RpiCamera
        return RpiCamera(CAM_W, CAM_H)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    return cap


def read_frame(cam):
    """Returns RGB (h, w, 3) uint8 or None on failure."""
    ok, bgr = cam.read()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def close_camera(cam):
    cam.release()


# ── cat image → binary edge mask ─────────────────────────────────────────────
def load_cat_mask(path: Path, w: int, h: int) -> np.ndarray:
    """
    Load the hand-drawn cat PNG (black lines on white background),
    return a (h, w) uint8 binary mask where lines = 255.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot find cat image at: {path.resolve()}")

    # black lines on white → invert so lines are bright
    inv = cv2.bitwise_not(img)

    # mild blur to fill tiny gaps in hand-drawn strokes, then threshold
    blurred = cv2.GaussianBlur(inv, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # resize to window
    resized = cv2.resize(binary, (w, h), interpolation=cv2.INTER_AREA)
    _, clean = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    return clean


# ── chamfer distance ──────────────────────────────────────────────────────────
def chamfer_distance(ref: np.ndarray, drawn: np.ndarray) -> float:
    """
    Average distance (px) from each drawn pixel to the nearest reference pixel.
    Lower = better trace.  Returns inf if canvas is blank.
    """
    ref_bin   = ref   > 128
    drawn_bin = drawn > 128
    if not drawn_bin.any():
        return float("inf")
    dist_map = distance_transform_edt(~ref_bin)
    return float(dist_map[drawn_bin].mean())


def score_label(d: float) -> str:
    if d == float("inf"): return "Nothing drawn yet"
    if d < 5:   return "Perfect!!"
    if d < 12:  return "Excellent!"
    if d < 22:  return "Great!"
    if d < 35:  return "Good"
    if d < 55:  return "Keep trying!"
    return "Way off — try again!"


# ── mask → pygame Surface (surfarray fast path) ───────────────────────────────
def mask_to_surface(mask: np.ndarray, rgb: tuple, alpha_max: int) -> pygame.Surface:
    h, w = mask.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)   # shape (w, h, 3), locked
    px[:, :, 0] = rgb[0]
    px[:, :, 1] = rgb[1]
    px[:, :, 2] = rgb[2]
    del px                                 # release surface lock
    al = pygame.surfarray.pixels_alpha(surf)  # shape (w, h)
    al[:, :] = (mask.T * (alpha_max / 255.0)).astype(np.uint8)
    del al                                 # release surface lock
    return surf


# ── dlib face detector + landmark predictor ───────────────────────────────────
# dlib 68-point model indices:
#   nose tip         = 30
#   inner upper lip  = 62
#   inner lower lip  = 66
#   left eye corner  = 36
#   right eye corner = 45
PREDICTOR_PATH = Path(__file__).parent / "shape_predictor_68_face_landmarks.dat"

def mouth_open(shape) -> bool:
    mouth_gap = abs(shape.part(66).y - shape.part(62).y)
    eye_dist  = abs(shape.part(45).x - shape.part(36).x)
    if eye_dist < 1:
        return False
    return (mouth_gap / eye_dist) > MOUTH_OPEN_RATIO


# ── main ──────────────────────────────────────────────────────────────────────
def run(screen=None, clock=None):
    """Cat Tracing Game.

    Master-launcher contract:
        run(screen, clock) -> "done" | "skip" | "quit"
    Reuses the passed-in screen/clock and never calls pygame.init()/quit().
    Called with no arguments it runs standalone (creates its own window).
    """
    embedded = screen is not None
    if not embedded:
        pygame.init()
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Cat Tracing — Nose Drawing")
    if clock is None:
        clock = pygame.time.Clock()

    # When launched by the master hub, grab it for the skip button/hotkey hooks.
    _m = sys.modules.get("master") if embedded else None

    font_lg  = pygame.font.SysFont("Arial", 26, bold=True)
    font_sm  = pygame.font.SysFont("Arial", 20)

    # ---- reference cat ----  (raise on failure; master's wrapper catches it)
    cat_ref  = load_cat_mask(CAT_IMAGE, WIN_W, WIN_H)
    cat_surf = mask_to_surface(cat_ref, CAT_TINT, OVERLAY_ALPHA)

    # ---- drawing canvas ----
    drawing   = np.zeros((WIN_H, WIN_W), dtype=np.uint8)
    draw_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    # ---- dlib face detector + shape predictor ----
    if not PREDICTOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing landmark model: {PREDICTOR_PATH}. Download with: "
            "wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
            "&& bzip2 -d shape_predictor_68_face_landmarks.dat.bz2"
        )
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

    # ---- camera ----
    cam = open_camera()

    prev_pt   = None
    score_str = None
    chamfer   = None

    result, running = "skip", True
    while running:
        # ── events ──
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                result, running = "quit", False
                break
            if _m is not None and _m.check_skip(ev):
                result, running = "skip", False
                break
            if ev.type == pygame.KEYDOWN:
                if _m is None and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break
                if ev.key == pygame.K_c:
                    drawing[:] = 0
                    draw_surf  = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                    score_str  = None
                    chamfer    = None
                    prev_pt    = None
                if ev.key == pygame.K_s:
                    chamfer   = chamfer_distance(cat_ref, drawing)
                    score_str = f"Chamfer: {chamfer:.1f} px  —  {score_label(chamfer)}"
        if not running:
            break

        # ── camera frame ──
        frame = read_frame(cam)
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)   # mirror so left/right feel natural

        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = detector(gray, 1)

        nose_pt = None
        is_open = False

        if faces:
            shape   = predictor(gray, faces[0])
            is_open = mouth_open(shape)
            sx = WIN_W / frame.shape[1]
            sy = WIN_H / frame.shape[0]
            nx = int(shape.part(30).x * sx)
            ny = int(shape.part(30).y * sy)
            nose_pt = (nx, ny)

            if is_open and prev_pt is not None:
                cv2.line(drawing, prev_pt, nose_pt, 255, LINE_THICK)
                pygame.draw.line(draw_surf, (*DRAW_COLOR, 220),
                                 prev_pt, nose_pt, LINE_THICK)
                score_str = None

            prev_pt = nose_pt if is_open else None

        else:
            prev_pt = None

        # ── render ──
        cam_bg   = cv2.resize(frame, (WIN_W, WIN_H))          # (h, w, 3) RGB
        cam_surf = pygame.image.frombuffer(
            cam_bg.tobytes(), (WIN_W, WIN_H), "RGB"
        )
        screen.blit(cam_surf,  (0, 0))   # live webcam background
        screen.blit(cat_surf,  (0, 0))   # ghost cat reference
        screen.blit(draw_surf, (0, 0))   # user strokes

        # nose cursor
        if nose_pt:
            col = CURSOR_OPEN if is_open else CURSOR_CLOSED
            pygame.draw.circle(screen, col, nose_pt, 10, 3)

        # ── HUD ──
        if score_str:
            hue = (80, 230, 80) if chamfer is not None and chamfer < 22 else (230, 120, 50)
            screen.blit(font_lg.render(score_str, True, hue), (10, 8))
        else:
            if nose_pt:
                tip_msg = "DRAWING — close mouth to lift pen" if is_open \
                          else "HOVERING — open mouth to draw"
                screen.blit(font_lg.render(tip_msg, True, (200, 200, 200)), (10, 8))
            else:
                screen.blit(font_lg.render("No face detected — look at the camera", True,
                                           (220, 100, 100)), (10, 8))

        status = f"{'[PEN DOWN]' if is_open else '[PEN UP  ]'}  nose at {nose_pt}" \
                 if nose_pt else "searching..."
        screen.blit(font_sm.render(status, True, (140, 200, 255)), (10, 40))
        screen.blit(font_sm.render("S = Score   C = Clear   ESC/SKIP = exit",
                                   True, (100, 100, 100)), (10, WIN_H - 28))

        if _m is not None:
            _m.draw_skip_button(screen)
        pygame.display.flip()
        clock.tick(60)

    # ── cleanup ──
    close_camera(cam)
    if not embedded:
        pygame.quit()
    return result


if __name__ == "__main__":
    run()