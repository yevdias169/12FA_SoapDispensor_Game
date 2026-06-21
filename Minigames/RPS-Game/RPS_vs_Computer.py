# RPS vs Computer
# Single-hand Rock Paper Scissor game against the computer.
# Press SPACE to start a round: computer randomly picks R/P/S,
# a countdown plays, then your hand gesture at "SHOOT" is captured and judged.
#
# Rendering: pygame (camera frames converted to pygame Surfaces each tick).
# Entry point: run() -> 'win' | 'quit'
#   'win'  – player won a round and pressed any key to continue
#   'quit' – Q/ESC pressed or window closed; master should exit cleanly

import cv2
import numpy as np
import os
import random
import sys
import time

import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared Pi-camera module lives in the Minigames/ directory (one level up).
_MINIGAMES_DIR = os.path.dirname(SCRIPT_DIR)
if _MINIGAMES_DIR not in sys.path:
    sys.path.insert(0, _MINIGAMES_DIR)

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition
import config


CHOICES = ['rock', 'paper', 'scissor']

GESTURE_TO_CHOICE = {
    'fist':  'rock',
    'five':  'paper',
    'three': 'scissor',
    'yeah':  'scissor',
}

COUNTDOWN_SEQUENCE = ['3', '2', '1', 'SHOOT!']
COUNTDOWN_STEP_SEC = 0.8
RESULT_DISPLAY_SEC = 3.0


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------

def judge(player, computer):
    if player == computer:
        return 'Tie'
    beats = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    return 'You win' if beats[player] == computer else 'Computer wins'


# ---------------------------------------------------------------------------
# Rendering helpers — identical style to Guessing_Game/game.py
# ---------------------------------------------------------------------------

def _to_surface(frame_bgr: np.ndarray, w: int, h: int) -> pygame.Surface:
    """Convert a BGR OpenCV frame to a scaled pygame Surface."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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


def _load_icon_surface(path: str, size: int) -> pygame.Surface:
    """Load a PNG icon (with alpha) as a pygame Surface at (size × size)."""
    surf = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(surf, (size, size))


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------

def _open_camera():
    """Open the camera source specified by config.CAMERA_BACKEND.

    Returns an object with .read() -> (ok, bgr_frame) and .release().
    The game loop mirrors frames itself, so the camera does not flip.
    """
    backend = getattr(config, "CAMERA_BACKEND", "opencv")
    if backend == "rpicam":
        from pi_camera import RpiCamera
        return RpiCamera(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    return cv2.VideoCapture(config.CAMERA_INDEX)


def _draw_hand_skeleton(img, disp, param):
    """Draw the hand skeleton onto the OpenCV frame before pygame conversion."""
    img_height, img_width, _ = img.shape
    for p in param:
        if p['class'] is None:
            continue
        for i in range(21):
            x = int(p['keypt'][i, 0])
            y = int(p['keypt'][i, 1])
            if 0 < x < img_width and 0 < y < img_height:
                start = p['keypt'][disp.ktree[i], :]
                x_ = int(start[0])
                y_ = int(start[1])
                if 0 < x_ < img_width and 0 < y_ < img_height:
                    cv2.line(img, (x_, y_), (x, y), disp.color[i], 2)
                cv2.circle(img, (x, y), 5, disp.color[i], -1)
    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(screen=None, clock=None):
    """Run the RPS minigame.

    Master-launcher contract:
        run(screen, clock) -> "done" | "skip" | "quit"
    Reuses the passed-in screen/clock and never calls pygame.init()/quit().

    Called with no arguments it runs standalone (creates its own window), so
    `python RPS_vs_Computer.py` still works unchanged.
    """
    embedded = screen is not None
    if not embedded:
        pygame.init()
        screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption(config.WINDOW_TITLE)
    if clock is None:
        clock = pygame.time.Clock()

    # When launched by the master hub, grab it for the skip button/hotkey hooks.
    _m = None
    if embedded:
        import sys as _sys
        _m = _sys.modules.get("master")

    font_huge   = pygame.font.SysFont("Arial", 88, bold=True)
    font_medium = pygame.font.SysFont("Arial", 36, bold=True)
    font_small  = pygame.font.SysFont("Arial", 24)

    pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)
    disp = DisplayHand(max_num_hands=1)
    cap  = _open_camera()
    gest = GestureRecognition(mode='eval')

    icons = {
        'rock':    _load_icon_surface(os.path.join(SCRIPT_DIR, 'images', 'Rockimage.png'),    config.ICON_SIZE),
        'paper':   _load_icon_surface(os.path.join(SCRIPT_DIR, 'images', 'Paperimage.png'),   config.ICON_SIZE),
        'scissor': _load_icon_surface(os.path.join(SCRIPT_DIR, 'images', 'Scissorimage.png'), config.ICON_SIZE),
    }

    W, H   = screen.get_size()
    cx, cy = W // 2, H // 2

    # State machine: 'wait' -> 'countdown' -> 'result' -> 'lose_prompt' | 'win_prompt' | 'wait'
    state           = 'wait'
    computer_choice = None
    player_choice   = None
    result_text     = None
    countdown_start = None
    result_start    = None
    prompt_start    = None
    live_choice     = None   # current live-detected RPS choice for top bar + icon

    result  = "done"
    running = True

    while running:
        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result = "quit"; running = False; break
            if _m is not None and _m.check_skip(event):
                result = "skip"; running = False; break

            if event.type == pygame.KEYDOWN:
                if _m is None and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False; break          # standalone quit

                elif state == 'wait' and event.key == pygame.K_SPACE:
                    computer_choice = random.choice(CHOICES)
                    state           = 'countdown'
                    countdown_start = time.time()

                elif state == 'win_prompt':
                    result  = "done"               # player won → finished
                    running = False; break
        if not running:
            break

        # ── Camera frame ─────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # Mediapipe — run gesture recognition on current frame
        frame.flags.writeable = False
        param = pipe.forward(frame)
        for p in param:
            if p['class'] is not None:
                p['gesture'] = gest.eval(p['angle'])
        frame.flags.writeable = True

        # Live choice for top bar and icon overlay
        p0 = param[0]
        live_choice = None
        if p0['class'] is not None:
            live_choice = GESTURE_TO_CHOICE.get(p0['gesture'])

        # Draw skeleton onto frame (baked in before pygame conversion)
        frame = _draw_hand_skeleton(frame, disp, param)

        # ── Render — frame → pygame surface ──────────────────────────────────
        screen.blit(_to_surface(frame, W, H), (0, 0))

        # Scale factors: frame pixel coords → pygame window coords
        fh, fw = frame.shape[:2]
        sx, sy = W / fw, H / fh

        # Live gesture icon near the wrist (pygame blit, scaled position)
        if p0['class'] is not None and live_choice is not None:
            ix = int((p0['keypt'][0, 0] - 30) * sx)
            iy = int((p0['keypt'][0, 1] + 40) * sy)
            screen.blit(icons[live_choice], (ix, iy))

        # ── Top info bar (always) ─────────────────────────────────────────────
        _overlay(screen, 0, 0, W, 64)
        # Reveal computer's choice only after the round resolves
        if state in ('result', 'lose_prompt', 'win_prompt') and computer_choice:
            comp_label = computer_choice.upper()
            comp_color = config.WHITE
        else:
            comp_label = '???'
            comp_color = config.GRAY
        _text(screen, f'Computer: {comp_label}', font_small, comp_color, 16, 18)
        hand_label = live_choice.upper() if live_choice else '—'
        hand_color = config.YELLOW if live_choice else config.GRAY
        _text(screen, f'Your hand: {hand_label}', font_small, hand_color, W - 240, 18)

        # ── State-specific rendering ──────────────────────────────────────────
        if state == 'countdown':
            elapsed = time.time() - countdown_start
            idx     = int(elapsed // COUNTDOWN_STEP_SEC)
            if idx < len(COUNTDOWN_SEQUENCE):
                cd_text = COUNTDOWN_SEQUENCE[idx]
                _overlay(screen, 0, cy - 80, W, 130)
                _text(screen, cd_text, font_huge, config.RED, cx, cy - 15, center=True)
            else:
                # SHOOT — capture gesture and judge
                player_choice = live_choice
                if player_choice is None:
                    result_text = 'No hand gesture detected'
                else:
                    result_text = judge(player_choice, computer_choice)
                state        = 'result'
                result_start = time.time()

        elif state == 'result':
            line1  = 'You: %s' % (player_choice.upper() if player_choice else 'NONE')
            line2  = 'Computer: %s' % computer_choice.upper()
            line3  = result_text.upper()
            color3 = (config.GREEN  if result_text == 'You win'        else
                      config.RED    if result_text == 'Computer wins'  else
                      config.YELLOW)

            _overlay(screen, 0, cy - 100, W, 200, alpha=200)
            _text(screen, line1, font_medium, config.WHITE, cx, cy - 55, center=True)
            _text(screen, line2, font_medium, config.WHITE, cx, cy - 10, center=True)
            _text(screen, line3, font_medium, color3,       cx, cy + 45, center=True)

            if time.time() - result_start > RESULT_DISPLAY_SEC:
                if result_text == 'You win':
                    state = 'win_prompt'
                elif result_text == 'Computer wins':
                    state        = 'lose_prompt'
                    prompt_start = time.time()
                else:  # Tie or no gesture
                    state = 'wait'

        elif state == 'lose_prompt':
            _overlay(screen, 0, cy - 100, W, 200, alpha=200)
            _text(screen, 'You suck, play again', font_medium, config.RED, cx, cy - 10, center=True)
            if time.time() - prompt_start > RESULT_DISPLAY_SEC:
                computer_choice = random.choice(CHOICES)
                state           = 'countdown'
                countdown_start = time.time()

        elif state == 'win_prompt':
            _overlay(screen, 0, cy - 100, W, 200, alpha=200)
            _text(screen, 'You win!', font_huge, config.GREEN, cx, cy - 55, center=True)
            _text(screen, 'Press any key to progress',
                  font_small, config.WHITE, cx, cy + 60, center=True)

        else:  # 'wait'
            _overlay(screen, 0, H - 54, W, 54)
            _text(screen, 'Press SPACE to play',
                  font_small, config.GRAY, cx, H - 27, center=True)

        if _m is not None:
            _m.draw_skip_button(screen)
        pygame.display.flip()
        clock.tick(config.FPS)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    pipe.pipe.close()
    cap.release()
    if not embedded:
        pygame.quit()
    return result


if __name__ == '__main__':
    run()
