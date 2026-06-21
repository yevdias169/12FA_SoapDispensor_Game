#! /usr/bin/env python3
"""
master.py — pygame launcher / menu hub for the minigame collection.

Running this opens ONE pygame window showing a menu of boxes (one per game,
plus an "All Games" box). The master owns the single pygame.init(), the one
display surface, the shared clock, and top-level quit handling.

Integration contract every minigame entry point conforms to:
    run(screen, clock) -> "done" | "skip" | "quit"
  - reuses the passed-in screen/clock; never calls pygame.init()/set_mode()/quit()
  - on pygame.QUIT  -> returns "quit"
  - for each event calls master.check_skip(event); if True -> returns "skip"
  - just before flipping, calls master.draw_skip_button(screen)

Only pygame is needed to launch this menu. Each game's heavy dependencies
(cv2, mediapipe, hsemotion, …) are imported lazily, inside its adapter, the
moment that game is launched — so the hub stays light and one game's missing
dependency only breaks that game, not the launcher.
"""

import os
import shutil
import sys
import traceback

import pygame

# Register this module under the name "master" so the minigames (which run in
# this same process) can do `sys.modules.get("master")` and reach the skip
# helpers below — even though we are executed as __main__.
sys.modules.setdefault("master", sys.modules[__name__])


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 960, 720
FPS = 60

# Auto-detect the camera backend: the Raspberry Pi ships the `rpicam-vid`
# binary (Pi Camera Module via pi_camera.RpiCamera); dev machines don't, so
# they fall back to a USB/built-in webcam. Override with the environment
# variable MINIGAME_CAMERA_BACKEND=opencv|rpicam.
CAMERA_BACKEND = os.environ.get("MINIGAME_CAMERA_BACKEND") or (
    "rpicam" if shutil.which("rpicam-vid") else "opencv")

FONT_NAME   = "Arial"
FONT_TITLE  = 48
FONT_BUTTON = 30
FONT_SMALL  = 22

# Colour palette (R, G, B)
BG          = ( 18,  18,  28)
BOX         = ( 45,  48,  70)
BOX_HOVER   = ( 80,  90, 140)
ALL_BOX     = ( 40,  90,  70)
ALL_HOVER   = ( 60, 140, 100)
BORDER      = (120, 130, 170)
WHITE       = (240, 240, 240)
GREY        = (150, 150, 160)
SKIP_BG     = ( 70,  40,  40)
SKIP_BORDER = (200, 120, 120)

# Skip hotkeys (the on-screen button is the requested control; these are the
# always-works fallback).
SKIP_KEYS = (pygame.K_ESCAPE, pygame.K_s)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MINIGAMES_DIR = os.path.join(REPO_ROOT, "Minigames")


def _front(path):
    """Force `path` to the front of sys.path (move it if already present)."""
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Font cache (pygame.font is only ready after pygame.init())
# ---------------------------------------------------------------------------

_FONT_CACHE = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = pygame.font.SysFont(FONT_NAME, size, bold=bold)
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# Master-provided skip helpers (called by every minigame loop)
# ---------------------------------------------------------------------------

_SKIP_RECT = None  # cached by draw_skip_button(), read by check_skip()


def draw_skip_button(screen):
    """Blit a small SKIP button in the top-right corner and cache its rect."""
    global _SKIP_RECT
    sw, _ = screen.get_size()
    w, h, margin = 96, 38, 12
    rect = pygame.Rect(sw - w - margin, margin, w, h)
    _SKIP_RECT = rect

    pygame.draw.rect(screen, SKIP_BG, rect, border_radius=6)
    pygame.draw.rect(screen, SKIP_BORDER, rect, width=2, border_radius=6)
    label = _font(FONT_SMALL, bold=True).render("SKIP", True, WHITE)
    screen.blit(label, label.get_rect(center=rect.center))


def check_skip(event):
    """True if the event is a click on the SKIP button or a skip-hotkey press."""
    if event.type == pygame.KEYDOWN and event.key in SKIP_KEYS:
        return True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if _SKIP_RECT is not None and _SKIP_RECT.collidepoint(event.pos):
            return True
    return False


# ---------------------------------------------------------------------------
# Camera helper (only the camera-based adapters use this)
# ---------------------------------------------------------------------------

def _open_camera():
    """Open a camera matching CAMERA_BACKEND; returns an object with
    .read() -> (ok, bgr) and .release()."""
    if CAMERA_BACKEND == "rpicam":
        _front(MINIGAMES_DIR)
        from pi_camera import RpiCamera
        return RpiCamera(640, 480)
    import cv2
    return cv2.VideoCapture(0)


# ---------------------------------------------------------------------------
# Minigame adapters — each conforms to run(screen, clock) -> str
#
# The three window-owning games (Guessing, RPS, rubiks) already conform after
# small edits, so their adapters just lazily import and delegate. The two embed
# games (Emotion, Flappy) are scene classes, driven here by a conforming loop.
#
# RPS and rubiks both `import config` by bare name from their own folders; that
# name would collide across games in one process, so we purge it and push the
# right folder to the front of sys.path before importing.
# ---------------------------------------------------------------------------

def run_guessing(screen, clock):
    _front(MINIGAMES_DIR)  # Guessing_Game is a package; its config is namespaced
    from Guessing_Game.game import run_game
    return run_game(screen, clock)


def run_rps(screen, clock):
    _front(MINIGAMES_DIR)
    _front(os.path.join(MINIGAMES_DIR, "RPS-Game"))
    sys.modules.pop("config", None)  # avoid sibling-game config collision
    import RPS_vs_Computer
    return RPS_vs_Computer.run(screen, clock)


def run_rubiks(screen, clock):
    _front(MINIGAMES_DIR)
    _front(os.path.join(MINIGAMES_DIR, "rubiks-vision"))
    sys.modules.pop("config", None)  # avoid sibling-game config collision
    from minigames.rubiks_checker import run_rubiks_checker
    return run_rubiks_checker(screen, clock)


def run_flappy(screen, clock):
    _front(os.path.join(MINIGAMES_DIR, "Flappy_Bird"))
    from flappybird_embed import FlappyBirdGame
    from flappybird_velocity import load_images, WIN_WIDTH, WIN_HEIGHT

    images = load_images()
    sw, sh = screen.get_size()
    rect = pygame.Rect((sw - WIN_WIDTH) // 2, (sh - WIN_HEIGHT) // 2,
                       WIN_WIDTH, WIN_HEIGHT)
    game = FlappyBirdGame(screen.subsurface(rect), images)

    result, running = "done", True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result, running = "quit", False
                break
            if check_skip(event):
                result, running = "skip", False
                break
            game.handle_event(event)
        if not running:
            break

        game.update()
        screen.fill(BG)        # clear the margins around the play area
        game.draw()
        draw_skip_button(screen)
        pygame.display.flip()

        if game.done:
            result, running = "done", False
    return result


def run_emotion(screen, clock):
    _front(os.path.join(MINIGAMES_DIR, "Emotion_Scanner"))
    from emotion_embed import EmotionGame
    import cv2

    cap = _open_camera()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # hsemotion_onnx's model downloader calls urllib.request.urlretrieve but only
    # does `import urllib` — importing the submodule here binds urllib.request
    # globally so that broken reference resolves (first-run model download only).
    import urllib.request  # noqa: F401
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")

    game = EmotionGame(screen, cap, face_cascade, recognizer)
    result, running = "done", True
    try:
        while running:
            dt = clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    result, running = "quit", False
                    break
                if check_skip(event):
                    result, running = "skip", False
                    break
                game.handle_event(event)
            if not running:
                break

            game.update(dt)
            game.draw()
            draw_skip_button(screen)
            pygame.display.flip()

            if game.done:
                result, running = "done", False
    finally:
        cap.release()
    return result


def run_cat(screen, clock):
    _front(MINIGAMES_DIR)
    _front(os.path.join(MINIGAMES_DIR, "cat"))
    from cat_tracing_game import run as run_cat_game
    return run_cat_game(screen, clock)


# ---------------------------------------------------------------------------
# Registry — (display name, callable). Order == "All Games" play order.
# ---------------------------------------------------------------------------

MINIGAMES = [
    ("Flappy Bird",       run_flappy),
    ("Emotion Scanner",   run_emotion),
    ("Finger Guessing",   run_guessing),
    ("Rock Paper Scissor", run_rps),
    ("Rubik's Checker",   run_rubiks),
    ("Cat Tracing",       run_cat),
]


# ---------------------------------------------------------------------------
# Menu layout + rendering
# ---------------------------------------------------------------------------

# State machine
MENU, PLAYING_SINGLE, PLAYING_SEQUENCE = "menu", "single", "sequence"

# Box kinds returned by hit-testing
KIND_GAME, KIND_ALL = "game", "all"

BOX_W, BOX_H = 280, 96
GAP = 24
TOP_MARGIN = 130


def _menu_layout(screen):
    """Compute the menu boxes as a list of (rect, label, kind, index).

    Auto-wrapping grid that fits the window width, so adding/removing games
    just reflows; nothing is hard-coded to a fixed game count.
    """
    sw = screen.get_width()
    entries = [(name, KIND_GAME, i) for i, (name, _) in enumerate(MINIGAMES)]
    entries.append(("All Games", KIND_ALL, -1))

    cols = max(1, (sw - GAP) // (BOX_W + GAP))
    cols = min(cols, len(entries))
    y0 = TOP_MARGIN

    boxes = []
    for idx, (label, kind, gi) in enumerate(entries):
        r, c = divmod(idx, cols)
        # Centre the last (possibly short) row.
        in_row = min(cols, len(entries) - r * cols)
        row_w = in_row * BOX_W + (in_row - 1) * GAP
        rx0 = (sw - row_w) // 2
        x = rx0 + c * (BOX_W + GAP)
        y = y0 + r * (BOX_H + GAP)
        boxes.append((pygame.Rect(x, y, BOX_W, BOX_H), label, kind, gi))
    return boxes


def _draw_menu(screen, boxes, hover_idx):
    screen.fill(BG)

    title = _font(FONT_TITLE, bold=True).render("MINIGAME HUB", True, WHITE)
    screen.blit(title, title.get_rect(center=(screen.get_width() // 2, 60)))

    hint = _font(FONT_SMALL).render(
        "Click a game to play it, or 'All Games' to run them in order  |  "
        "ESC/S or the SKIP button skips a game",
        True, GREY)
    screen.blit(hint, hint.get_rect(center=(screen.get_width() // 2, 100)))

    for i, (rect, label, kind, _gi) in enumerate(boxes):
        hovered = (i == hover_idx)
        if kind == KIND_ALL:
            color = ALL_HOVER if hovered else ALL_BOX
        else:
            color = BOX_HOVER if hovered else BOX
        pygame.draw.rect(screen, color, rect, border_radius=10)
        pygame.draw.rect(screen, BORDER, rect, width=2, border_radius=10)

        txt = _font(FONT_BUTTON, bold=(kind == KIND_ALL)).render(label, True, WHITE)
        screen.blit(txt, txt.get_rect(center=rect.center))


# ---------------------------------------------------------------------------
# Launch wrapper — isolates a buggy game so it can't take down the hub
# ---------------------------------------------------------------------------

def _launch(fn, screen, clock):
    """Run one minigame callable. Returns "done"/"skip"/"quit".

    Any exception is logged and treated as "done" (return to menu) so one
    broken game never crashes the launcher.
    """
    try:
        result = fn(screen, clock)
    except Exception:
        print(f"\n[master] Minigame '{fn.__name__}' crashed:", file=sys.stderr)
        traceback.print_exc()
        return "done"
    return result if result in ("done", "skip", "quit") else "done"


# ---------------------------------------------------------------------------
# Main loop / state machine
# ---------------------------------------------------------------------------

def main():
    # --all : kiosk mode — skip the menu, run every game in order, then exit
    # (returns control to whatever launched master.py, e.g. the web app).
    auto_all = "--all" in sys.argv

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Minigame Hub")
    clock = pygame.time.Clock()

    state = PLAYING_SEQUENCE if auto_all else MENU
    seq_idx = 0
    running = True

    while running:
        if state == MENU:
            boxes = _menu_layout(screen)
            mouse = pygame.mouse.get_pos()
            hover_idx = next(
                (i for i, (rect, *_r) in enumerate(boxes) if rect.collidepoint(mouse)),
                None,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, _label, kind, gi in boxes:
                        if rect.collidepoint(event.pos):
                            if kind == KIND_ALL:
                                state, seq_idx = PLAYING_SEQUENCE, 0
                            else:
                                state, seq_idx = PLAYING_SINGLE, gi
                            break

            _draw_menu(screen, boxes, hover_idx)
            pygame.display.flip()
            clock.tick(FPS)

        elif state == PLAYING_SINGLE:
            result = _launch(MINIGAMES[seq_idx][1], screen, clock)
            if result == "quit":
                running = False
            else:               # "done" or "skip" -> back to menu
                state = MENU

        elif state == PLAYING_SEQUENCE:
            result = _launch(MINIGAMES[seq_idx][1], screen, clock)
            if result == "quit":
                running = False
            else:               # "done" or "skip" -> advance to next game
                seq_idx += 1
                if seq_idx >= len(MINIGAMES):
                    # In kiosk (--all) mode, exit when the sequence finishes so
                    # control returns to the launcher; otherwise back to menu.
                    if auto_all:
                        running = False
                    else:
                        state = MENU

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
