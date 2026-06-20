"""
rubiks_checker.py — pygame scene that verifies whether a Rubik's Cube is solved.

Integration contract (master-loop snippet):
────────────────────────────────────────────
    scene = RubiksCheckerScene(screen)
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        events = pygame.event.get()
        scene.handle_events(events)
        scene.update(dt)
        scene.draw(screen)
        pygame.display.flip()
        if scene.finished:
            running = False
    result = scene.result
    scene.close()   # frees the camera so the master menu / next minigame can use it
────────────────────────────────────────────

Lifecycle: camera opened in __init__, always released in close() (idempotent).
"""

from __future__ import annotations

import enum
import os
import sys

import numpy as np
import pygame

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from src.camera import create_camera
from src.frame_grabber import FrameGrabber
from src.face_detector import FaceDetector
from src.sticker_sampler import StickerSampler
from src.color_classifier import load_palette, create_classifier
from src.verifier import (
    Pipeline, CubeCheckerFSM, FaceResult,
    Verdict, check_cube,
)
from src.ui.render import cv2_to_surface, fit_frame, draw_sticker_grid, draw_guide_box
from src.ui.hud import HUD
from src.ui.toast import FaceToast
from src.ui.final_overlay import FinalOverlay


# ---------------------------------------------------------------------------
# Result enum
# ---------------------------------------------------------------------------

class RubiksResult(enum.Enum):
    SOLVED  = "SOLVED"
    ABORTED = "ABORTED"
    FAILED  = "FAILED"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class RubiksCheckerScene:
    """
    Drop-in pygame scene for Rubik's Cube face-by-face verification.
    The master loop calls handle_events / update / draw each frame.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        cfg=None,
        on_complete=None,
    ) -> None:
        self._cfg = cfg or config
        self._on_complete = on_complete
        self._screen_w, self._screen_h = screen.get_size()

        # Camera + threaded grabber
        self._camera  = create_camera(self._cfg)
        self._grabber = FrameGrabber(self._camera)

        # Vision pipeline
        detector = FaceDetector(
            mode=self._cfg.DETECTION_MODE,
            roi_frac=self._cfg.ROI_FRAC,
            warp_size=self._cfg.WARP_SIZE,
            min_face_area_frac=self._cfg.MIN_FACE_AREA_FRAC,
        )
        sampler = StickerSampler(
            warp_size=self._cfg.WARP_SIZE,
            patch_frac=self._cfg.PATCH_FRAC,
            glare_v_thresh=self._cfg.GLARE_V_THRESH,
            glare_s_thresh=self._cfg.GLARE_S_THRESH,
        )
        palette    = load_palette(self._cfg.CALIBRATION_PATH, self._cfg.DEFAULT_PALETTE)
        classifier = create_classifier(
            self._cfg.CLASSIFIER, palette,
            self._cfg.DELTA_E_MAX, self._cfg.HSV_PALETTE,
        )
        self._pipeline = Pipeline(
            detector, sampler, classifier,
            self._cfg.INTRA_FACE_DELTA_E, self._cfg.MAX_UNCERTAIN,
        )
        self._detector = detector   # kept for guide_rect()

        # Session FSM
        self._fsm = CubeCheckerFSM(
            stable_frames=self._cfg.STABLE_FRAMES,
            stable_clear_frames=self._cfg.STABLE_CLEAR_FRAMES,
            expected_colors=self._cfg.EXPECTED_COLORS,
        )

        # UI — fonts allocated once
        self._hud        = HUD(self._screen_w, self._screen_h, self._cfg.EXPECTED_COLORS)
        self._font_small = pygame.font.SysFont("Arial", 14)
        self._font_big   = pygame.font.SysFont("Arial", 28, bold=True)
        self._font_mid   = pygame.font.SysFont("Arial", 18)

        # Toast / overlay (created fresh each time they are needed)
        self._toast:   FaceToast    | None = None
        self._overlay: FinalOverlay | None = None

        # Runtime state
        self._state: str = "SCANNING"
        self._last_result: FaceResult = FaceResult()
        self._frame_surf: pygame.Surface | None = None
        self._failed_reason: str = ""
        self._finished: bool = False
        self._result: RubiksResult = RubiksResult.ABORTED
        self._closed: bool = False
        # Set True when the 6th face is captured so FACE_VERIFIED skips the
        # clear-debounce and goes straight to ALL_DONE after the toast.
        self._final_face_pending: bool = False

        # Pre-render the "waiting" message so it isn't created per-frame
        font_wait = pygame.font.SysFont("Arial", 24)
        self._waiting_surf = font_wait.render("Waiting for camera…", True, (180, 180, 180))

    # ------------------------------------------------------------------
    # Scene interface
    # ------------------------------------------------------------------

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        for ev in events:
            if ev.type == pygame.QUIT:
                self._abort()
                return
            if ev.type == pygame.KEYDOWN:
                self._handle_key(ev.key)

    def update(self, dt: float) -> None:
        if self._finished:
            return

        # ── grab latest camera frame ──────────────────────────────────
        frame = self._grabber.read_latest()
        if frame is not None:
            result = self._pipeline.evaluate(frame)
            self._last_result = result
            self._frame_surf  = cv2_to_surface(frame)
        else:
            result = self._last_result

        # ── advance animation clocks ──────────────────────────────────
        if self._toast:
            self._toast.update(dt)
        if self._overlay:
            self._overlay.update(dt)

        # ── FSM ticks ────────────────────────────────────────────────
        if self._state == "SCANNING":
            for ev in self._fsm.feed(result):
                if ev.startswith(CubeCheckerFSM.EV_FACE_VERIFIED):
                    self._enter_face_verified(ev.split(":")[1])
                if ev == CubeCheckerFSM.EV_ALL_DONE:
                    # Completion detected at capture time — skip clear-debounce
                    self._final_face_pending = True

        elif self._state == "FACE_VERIFIED":
            toast_done = self._toast is None or self._toast.is_done
            if self._final_face_pending:
                # Last face: go straight to ALL_DONE once the toast finishes;
                # do NOT require the clear-debounce (no "face 7 of 6" path).
                if toast_done:
                    self._final_face_pending = False
                    self._enter_all_done()
            else:
                self._fsm.tick_clear_debounce(result)
                if toast_done and self._fsm.clear_ready:
                    for ev in self._fsm.advance_from_face_verified():
                        if ev == CubeCheckerFSM.EV_ALL_DONE:
                            self._enter_all_done()
                        elif ev == CubeCheckerFSM.EV_BACK_SCANNING:
                            self._state = "SCANNING"

        elif self._state == "CUBE_VERIFIED":
            if self._overlay and self._overlay.is_done:
                self._result   = RubiksResult.SOLVED
                self._finished = True
                if self._on_complete:
                    self._on_complete(self._result)
                if self._cfg.TERMINATE_ON_COMPLETE:
                    self.close()
                    pygame.quit()
                    sys.exit(0)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((0, 0, 0))

        # ── video feed ────────────────────────────────────────────────
        if self._frame_surf is not None:
            scaled, ox, oy, scale = fit_frame(
                self._frame_surf, self._screen_w, self._screen_h
            )
            surface.blit(scaled, (ox, oy))

            draw_sticker_grid(
                surface, self._last_result,
                self._cfg.WARP_SIZE, scale, ox, oy,
                self._font_small,
            )
            # Guide box (build a dummy zero-frame just to get dimensions)
            guide_rect = self._detector.guide_rect(
                np.zeros(
                    (self._cfg.FRAME_HEIGHT, self._cfg.FRAME_WIDTH, 3),
                    dtype=np.uint8,
                )
            )
            draw_guide_box(surface, guide_rect, scale, ox, oy)
        else:
            surface.blit(
                self._waiting_surf,
                ((self._screen_w - self._waiting_surf.get_width()) // 2,
                 self._screen_h // 2),
            )

        # ── HUD ───────────────────────────────────────────────────────
        if self._state in ("SCANNING", "FACE_VERIFIED"):
            n_captured = len(self._fsm.captured)
            n_expected = len(self._cfg.EXPECTED_COLORS)
            # Clamp so we never pass face_num >= total_faces to the HUD.
            # When all faces are captured the prompt is suppressed inside HUD.draw().
            self._hud.draw(
                surface,
                self._last_result,
                self._fsm.captured,
                self._fsm.tracker.count,
                self._cfg.STABLE_FRAMES,
                self._fsm.duplicate_hint,
                min(n_captured, n_expected),   # clamped face_num
                n_expected,
            )

        # ── Toast ─────────────────────────────────────────────────────
        if self._toast and not self._toast.is_done:
            self._toast.draw(surface)

        # ── Final overlay ─────────────────────────────────────────────
        if self._overlay and not self._overlay.is_done:
            self._overlay.draw(surface)

        # ── Failed screen ─────────────────────────────────────────────
        if self._state == "FAILED_SCREEN":
            self._draw_failed_screen(surface)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def result(self) -> RubiksResult:
        return self._result

    def close(self) -> None:
        """Release camera + capture thread. Safe to call multiple times."""
        if not self._closed:
            self._closed = True
            self._grabber.stop()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _enter_face_verified(self, color: str) -> None:
        self._state = "FACE_VERIFIED"
        self._toast = FaceToast(
            color_label=color,
            screen_w=self._screen_w,
            screen_h=self._screen_h,
            t_in=self._cfg.TOAST_IN,
            t_hold=self._cfg.TOAST_HOLD,
            t_out=self._cfg.TOAST_OUT,
        )

    def _enter_all_done(self) -> None:
        face_results = list(self._fsm.captured.values())
        cube_result  = check_cube(face_results)
        if cube_result.verdict == Verdict.SOLVED:
            self._state   = "CUBE_VERIFIED"
            self._overlay = FinalOverlay(
                screen_w=self._screen_w,
                screen_h=self._screen_h,
                backdrop_fade=self._cfg.FINAL_BACKDROP_FADE,
                pop_time=self._cfg.FINAL_POP_TIME,
                auto_dismiss=self._cfg.FINAL_AUTO_DISMISS,
            )
        else:
            dups  = cube_result.duplicate_colors
            fails = cube_result.failed_faces
            if dups:
                self._failed_reason = (
                    f"Duplicate face colour(s): {', '.join(dups)}. "
                    "Recalibrate or rescan."
                )
            elif fails:
                self._failed_reason = (
                    f"Face(s) {[f + 1 for f in fails]} failed uniform check."
                )
            else:
                self._failed_reason = "Cube verification failed."
            self._state = "FAILED_SCREEN"

    def _abort(self) -> None:
        self._result   = RubiksResult.ABORTED
        self._finished = True
        self.close()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self._abort()
            return

        if key == pygame.K_r:
            self._fsm.reset_session()
            self._state   = "SCANNING"
            self._toast   = None
            self._overlay = None
            return

        if key == pygame.K_SPACE and self._state == "SCANNING":
            for ev in self._fsm.force_verify(self._last_result):
                if ev.startswith(CubeCheckerFSM.EV_FACE_VERIFIED):
                    self._enter_face_verified(ev.split(":")[1])
                    break

        if self._state == "CUBE_VERIFIED" and self._overlay:
            self._overlay.dismiss()

    # ------------------------------------------------------------------
    # Failed screen
    # ------------------------------------------------------------------

    def _draw_failed_screen(self, surface: pygame.Surface) -> None:
        bg = pygame.Surface((self._screen_w, self._screen_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        surface.blit(bg, (0, 0))

        title  = self._font_big.render("VERIFICATION FAILED", True, (220, 60, 60))
        reason = self._font_mid.render(self._failed_reason,        True, (210, 210, 210))
        hint   = self._font_mid.render(
            "Press R to retry   |   ESC to quit", True, (160, 160, 160)
        )
        cy = self._screen_h // 2 - 60
        for i, s in enumerate((title, reason, hint)):
            surface.blit(s, ((self._screen_w - s.get_width()) // 2, cy + i * 44))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_rubiks_checker(
    screen: pygame.Surface | None = None,
    cfg=None,
) -> RubiksResult:
    """
    Creates its own window if screen is None, runs an internal pygame loop
    until the scene finishes, then returns the result.
    Releases the camera before returning so the caller can reuse the device.
    """
    cfg = cfg or config
    own_window = screen is None

    if own_window:
        pygame.init()
        screen = pygame.display.set_mode(cfg.WINDOW_SIZE)
        pygame.display.set_caption("Rubik's Cube Verifier")

    clock = pygame.time.Clock()
    scene = RubiksCheckerScene(screen, cfg=cfg)

    running = True
    while running:
        dt = clock.tick(cfg.FPS) / 1000.0
        scene.handle_events(pygame.event.get())
        scene.update(dt)
        scene.draw(screen)
        pygame.display.flip()
        if scene.finished:
            running = False

    result = scene.result
    scene.close()

    if own_window:
        pygame.quit()

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_rubiks_checker()
    print(f"Result: {result.value}")
