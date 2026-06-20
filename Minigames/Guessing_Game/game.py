"""
Finger Guessing Game — pygame-based game loop.

UI decision: pygame (not OpenCV imshow) so this module slots into the
future pygame master controller without any changes to the rendering layer.
OpenCV frames are converted to pygame Surfaces each tick.

Guess trigger: SPACE key locks in the current live prediction as the
official guess. The model runs continuously while waiting so the user sees
live feedback, but jitter does not count — only the moment SPACE is pressed.
"""

import random
from enum import Enum, auto

import cv2
import numpy as np
import pygame

from . import config
from .model_wrapper import ModelWrapper


class _State(Enum):
    WAITING = auto()  # live feed + live prediction preview; SPACE to guess
    RESULT  = auto()  # show correct/incorrect; any key to continue


def _to_surface(frame_bgr: np.ndarray, w: int, h: int) -> pygame.Surface:
    """Convert a BGR OpenCV frame to a scaled pygame Surface."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # OpenCV shape is (H, W, C); pygame.surfarray.make_surface expects (W, H, C)
    surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    return pygame.transform.scale(surf, (w, h))


def _overlay(screen: pygame.Surface, x: int, y: int, w: int, h: int, alpha: int = 160):
    """Draw a semi-transparent black rectangle."""
    bar = pygame.Surface((w, h))
    bar.set_alpha(alpha)
    bar.fill((0, 0, 0))
    screen.blit(bar, (x, y))


def _text(screen, text, font, color, x, y, *, center=False):
    """Render text with a 2-px drop shadow for legibility over camera feed."""
    shadow = font.render(text, True, (0, 0, 0))
    sr = shadow.get_rect()
    if center:
        sr.center = (x + 2, y + 2)
    else:
        sr.topleft = (x + 2, y + 2)
    screen.blit(shadow, sr)

    surf = font.render(text, True, color)
    r = surf.get_rect()
    if center:
        r.center = (x, y)
    else:
        r.topleft = (x, y)
    screen.blit(surf, r)


class _PiCamera2:
    """
    Thin picamera2 wrapper with a cv2.VideoCapture-compatible interface
    (read() → (ok, bgr_frame), release()).
    picamera2 is imported lazily so this file remains importable on macOS.
    """

    def __init__(self, width: int, height: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise ImportError(
                "picamera2 is not installed. On Raspberry Pi: "
                "sudo apt install -y python3-picamera2"
            ) from exc
        self._cam = Picamera2()
        cam_cfg = self._cam.create_video_configuration(
            main={"format": "XRGB8888", "size": (width, height)}
        )
        self._cam.configure(cam_cfg)
        self._cam.start()

    def read(self) -> tuple[bool, np.ndarray]:
        # picamera2 returns XRGB (4 ch, channel 0 is padding); convert to BGR
        frame = self._cam.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    def release(self) -> None:
        self._cam.stop()


def _open_camera():
    """Open the camera source specified by config.CAMERA_BACKEND.

    Returns an object with .read() -> (ok, bgr_frame) and .release().
    """
    backend = getattr(config, "CAMERA_BACKEND", "opencv")
    if backend == "picamera2":
        return _PiCamera2(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    # opencv path — try configured index, then fall back
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        return cap
    fallback = 0 if config.CAMERA_INDEX != 0 else 1
    cap = cv2.VideoCapture(fallback)
    if cap.isOpened():
        return cap
    raise RuntimeError(
        f"Cannot open camera at index {config.CAMERA_INDEX} or {fallback}. "
        "Grant Terminal camera permission in System Settings → Privacy & Security → Camera."
    )


def run_game():
    """
    Entry point for the Finger Guessing Game.

    The pygame master controller should import and call this function:
        from Guessing_Game import run_game
        run_game()

    Returns cleanly when the game ends. Does NOT call sys.exit() so the
    parent process keeps running.
    """
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption(config.WINDOW_TITLE)
    clock = pygame.time.Clock()

    font_huge   = pygame.font.SysFont("Arial", 88, bold=True)
    font_medium = pygame.font.SysFont("Arial", 36, bold=True)
    font_small  = pygame.font.SysFont("Arial", 24)

    # Load model once at startup — not inside the per-frame loop
    model = ModelWrapper()
    cap   = _open_camera()

    W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
    cx, cy = W // 2, H // 2

    target           = random.randint(1, 5)
    state            = _State.WAITING
    live_prediction  = None   # updated every frame while WAITING
    locked_prediction = None  # set on SPACE press
    correct          = False

    running = True
    while running:
        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif state == _State.WAITING and event.key == pygame.K_SPACE:
                    if live_prediction is None:
                        pass  # no hand detected; stay — user sees "No hand" warning
                    else:
                        locked_prediction = live_prediction
                        correct = (locked_prediction == target)
                        state   = _State.RESULT

                elif state == _State.RESULT:
                    if correct:
                        running = False          # correct → any key exits
                    else:
                        # incorrect → retry the same target
                        state             = _State.WAITING
                        live_prediction   = None
                        locked_prediction = None

        # ── Camera frame ─────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror horizontally — matches test_classifier.py line 37
        frame = cv2.flip(frame, 1)

        # Run live prediction every frame while in WAITING state
        if state == _State.WAITING:
            live_prediction = model.predict(frame)

        # ── Render ───────────────────────────────────────────────────────────
        screen.blit(_to_surface(frame, W, H), (0, 0))

        # Top info bar
        _overlay(screen, 0, 0, W, 80)
        _text(screen, f"Target: {target} finger{'s' if target != 1 else ''}",
              font_medium, config.YELLOW, 16, 18)
        pred_str = str(live_prediction) if live_prediction is not None else "—"
        _text(screen, f"Showing: {pred_str}",
              font_medium, config.WHITE, W - 220, 18)

        if state == _State.WAITING:
            # Bottom instruction bar
            _overlay(screen, 0, H - 54, W, 54)
            if live_prediction is None:
                msg = "No hand detected — hold a hand in view, then press SPACE  |  Q to quit"
                color = config.RED
            else:
                msg = "Press SPACE to lock in your guess  |  Q to quit"
                color = config.GRAY
            _text(screen, msg, font_small, color, cx, H - 27, center=True)

        else:  # _State.RESULT
            # Centre result banner
            _overlay(screen, 0, cy - 100, W, 200, alpha=200)

            if correct:
                _text(screen, "Correct!", font_huge, config.GREEN, cx, cy - 55, center=True)
                _text(screen, "Press any key to quit", font_small, config.WHITE, cx, cy + 60, center=True)
            else:
                _text(screen, "Incorrect", font_huge, config.RED, cx, cy - 55, center=True)
                _text(screen,
                      f"You showed {locked_prediction},  target was {target}  —  press any key to try again  |  Q to quit",
                      font_small, config.WHITE, cx, cy + 60, center=True)

        pygame.display.flip()
        clock.tick(config.FPS)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    model.close()
    pygame.quit()
    # Intentionally no sys.exit() — caller (master controller) keeps running
