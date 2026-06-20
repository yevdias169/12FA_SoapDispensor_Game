"""
hud.py — top-left live status panel + bottom face-progress tracker.

All surfaces are allocated once in __init__; nothing is created per-frame.
"""

from __future__ import annotations

import pygame

from src.verifier import FaceStatus, FaceResult


# Palette label → RGB colour for the swatch squares
_SWATCH_RGB: dict[str, tuple[int, int, int]] = {
    "white":  (255, 255, 255),
    "yellow": (255, 220, 0  ),
    "red":    (210, 30,  30 ),
    "orange": (255, 130, 0  ),
    "green":  (30,  180, 30 ),
    "blue":   (30,  80,  210),
}

_STATUS_COLOR: dict[FaceStatus, tuple[int, int, int]] = {
    FaceStatus.UNIFORM:     (0,   210, 60 ),
    FaceStatus.NON_UNIFORM: (210, 80,  0  ),
    FaceStatus.NO_FACE:     (130, 130, 130),
}

_STATUS_LABEL: dict[FaceStatus, str] = {
    FaceStatus.UNIFORM:     "UNIFORM",
    FaceStatus.NON_UNIFORM: "NON-UNIFORM",
    FaceStatus.NO_FACE:     "NO FACE",
}


class HUD:
    """Draws the live status line and the six-colour progress tracker."""

    SWATCH_SIZE = 32
    SWATCH_GAP  = 8

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        expected_colors: list[str],
    ) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._expected = expected_colors

        self._font_status  = pygame.font.SysFont("Arial", 22, bold=True)
        self._font_label   = pygame.font.SysFont("Arial", 13)
        self._font_hint    = pygame.font.SysFont("Arial", 17)
        self._font_instr   = pygame.font.SysFont("Arial", 15)

        # Pre-render static instruction text
        self._instr = self._font_instr.render(
            "Hold face steady to auto-verify  |  SPACE = manual  |  R = reset  |  ESC = quit",
            True, (180, 180, 180),
        )

        self._check_surf = self._font_status.render("✓", True, (0, 210, 60))

    # ------------------------------------------------------------------
    # Public draw
    # ------------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        result: FaceResult,
        captured: dict[str, object],  # color → FaceResult
        tracker_count: int,
        stable_frames: int,
        duplicate_hint: str | None,
        face_num: int,
        total_faces: int,
    ) -> None:
        self._draw_status_panel(surface, result, tracker_count, stable_frames)
        self._draw_progress(surface, captured)
        self._draw_instructions(surface)
        if duplicate_hint:
            self._draw_duplicate_hint(surface, duplicate_hint)
        elif face_num < total_faces:
            # Suppress the "Show face N of M" prompt once all faces are captured
            # (face_num is already clamped by the scene to avoid "7 of 6")
            self._draw_face_prompt(surface, face_num, total_faces)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _draw_status_panel(
        self,
        surface: pygame.Surface,
        result: FaceResult,
        count: int,
        stable_frames: int,
    ) -> None:
        """Top-left: status label + stability progress bar."""
        status = result.status
        color = _STATUS_COLOR.get(status, (130, 130, 130))
        label = _STATUS_LABEL.get(status, "?")

        # Dark background panel
        panel = pygame.Surface((260, 56), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        surface.blit(panel, (8, 8))

        # Status text
        txt = self._font_status.render(label, True, color)
        _blit_outlined(surface, txt, (14, 14))

        # Colour label when UNIFORM
        if status == FaceStatus.UNIFORM and result.color_label:
            cl = self._font_label.render(
                result.color_label.upper(), True, (220, 220, 220)
            )
            surface.blit(cl, (14 + txt.get_width() + 8, 20))

        # Stability bar
        bar_x, bar_y = 14, 44
        bar_w, bar_h = 220, 8
        filled = int(bar_w * min(count, stable_frames) / stable_frames)
        pygame.draw.rect(surface, (50, 50, 50),   (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(surface, color,           (bar_x, bar_y, filled, bar_h))
        pygame.draw.rect(surface, (140, 140, 140), (bar_x, bar_y, bar_w, bar_h), 1)

    def _draw_progress(
        self,
        surface: pygame.Surface,
        captured: dict[str, object],
    ) -> None:
        """Bottom: six colour swatches — filled=captured, dim=remaining."""
        n = len(self._expected)
        total_w = n * self.SWATCH_SIZE + (n - 1) * self.SWATCH_GAP
        start_x = (self._screen_w - total_w) // 2
        y = self._screen_h - self.SWATCH_SIZE - 14

        for i, color in enumerate(self._expected):
            x = start_x + i * (self.SWATCH_SIZE + self.SWATCH_GAP)
            rgb = _SWATCH_RGB.get(color, (100, 100, 100))
            rect = pygame.Rect(x, y, self.SWATCH_SIZE, self.SWATCH_SIZE)

            if color in captured:
                pygame.draw.rect(surface, rgb, rect)
                pygame.draw.rect(surface, (255, 255, 255), rect, 2)
                # Checkmark
                c = self._font_label.render("✓", True, (0, 0, 0))
                surface.blit(c, (x + (self.SWATCH_SIZE - c.get_width()) // 2,
                                 y + (self.SWATCH_SIZE - c.get_height()) // 2))
            else:
                # Dim outline only
                dim = tuple(max(30, v // 4) for v in rgb)
                pygame.draw.rect(surface, dim, rect)
                pygame.draw.rect(surface, rgb, rect, 2)

            lbl = self._font_label.render(color[:3], True, (200, 200, 200))
            surface.blit(lbl, (x + (self.SWATCH_SIZE - lbl.get_width()) // 2,
                               y - lbl.get_height() - 2))

    def _draw_instructions(self, surface: pygame.Surface) -> None:
        surface.blit(
            self._instr,
            (self._screen_w - self._instr.get_width() - 10,
             self._screen_h - self._instr.get_height() - 55),
        )

    def _draw_duplicate_hint(
        self, surface: pygame.Surface, color: str
    ) -> None:
        msg = self._font_hint.render(
            f"Already scanned {color.upper()} — show a different face",
            True, (255, 165, 0),
        )
        _blit_outlined(surface, msg,
                       ((self._screen_w - msg.get_width()) // 2,
                        self._screen_h - 95))

    def _draw_face_prompt(
        self,
        surface: pygame.Surface,
        face_num: int,
        total_faces: int,
    ) -> None:
        remaining = total_faces - face_num
        msg = self._font_hint.render(
            f"Show face {face_num + 1} of {total_faces}  ({remaining} remaining)",
            True, (200, 200, 200),
        )
        surface.blit(
            msg,
            ((self._screen_w - msg.get_width()) // 2,
             self._screen_h - 95),
        )


def _blit_outlined(
    surface: pygame.Surface,
    text_surf: pygame.Surface,
    pos: tuple[int, int],
) -> None:
    """Blit text with a 1-px dark outline for readability over video."""
    ox, oy = pos
    # Outline
    shadow = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 0))
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        surface.blit(text_surf, (ox + dx, oy + dy))
    # Foreground (re-blit original on top would overwrite outline — blit outline copies then fg)
    surface.blit(text_surf, pos)
