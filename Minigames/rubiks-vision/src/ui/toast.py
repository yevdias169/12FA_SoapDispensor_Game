"""
toast.py — "{COLOR} FACE VERIFIED" animated popup.

Timeline (all dt-based, FPS-independent):
  0 … TOAST_IN          : fade + slide-in (alpha 0→255, rises ~20 px)
  TOAST_IN … +TOAST_HOLD: hold at full opacity
  end … TOAST_OUT       : fade-out

The panel has a left accent bar painted in the actual verified sticker colour,
a check glyph, and bold text.
"""

from __future__ import annotations

import pygame


# Palette label → RGB for the accent bar
_LABEL_RGB: dict[str, tuple[int, int, int]] = {
    "white":  (240, 240, 240),
    "yellow": (255, 215, 0  ),
    "red":    (210, 30,  30 ),
    "orange": (255, 130, 0  ),
    "green":  (30,  180, 30 ),
    "blue":   (30,  80,  210),
}


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class FaceToast:
    """
    Single-use animated toast.  Call draw(surface, dt) every frame.
    Check is_done to know when to discard it.
    """

    PANEL_W = 420
    PANEL_H = 70
    SLIDE_PX = 20      # vertical slide distance during fade-in

    def __init__(
        self,
        color_label: str,
        screen_w: int,
        screen_h: int,
        t_in: float,
        t_hold: float,
        t_out: float,
    ) -> None:
        self._color_label = color_label
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._t_in   = t_in
        self._t_hold = t_hold
        self._t_out  = t_out
        self._total  = t_in + t_hold + t_out
        self._elapsed = 0.0
        self._done = False

        accent_rgb = _LABEL_RGB.get(color_label, (180, 180, 180))

        # Build the panel surface once (no alpha yet — we apply per-frame)
        self._panel = pygame.Surface((self.PANEL_W, self.PANEL_H), pygame.SRCALPHA)
        self._panel.fill((20, 20, 20, 230))
        # Left accent bar (8 px wide)
        pygame.draw.rect(self._panel, (*accent_rgb, 255), (0, 0, 8, self.PANEL_H))

        # Fonts
        font_big   = pygame.font.SysFont("Arial", 26, bold=True)
        font_check = pygame.font.SysFont("Arial", 30, bold=True)

        label_text = f"{color_label.upper()} FACE VERIFIED"
        self._label_surf  = font_big.render(label_text,  True, (255, 255, 255))
        self._check_surf  = font_check.render("✓ ",      True, (0,   210, 60))

        # Pre-compose text into the panel at correct positions
        check_x = 16
        check_y = (self.PANEL_H - self._check_surf.get_height()) // 2
        label_x = check_x + self._check_surf.get_width() + 4
        label_y = (self.PANEL_H - self._label_surf.get_height()) // 2
        self._panel.blit(self._check_surf, (check_x, check_y))
        self._panel.blit(self._label_surf, (label_x,  label_y))

        self._panel_x = (screen_w - self.PANEL_W) // 2

    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the animation clock. Call from scene.update(dt)."""
        if not self._done:
            self._elapsed += dt
            if self._elapsed >= self._total:
                self._done = True

    def draw(self, surface: pygame.Surface) -> None:
        """Render the current animation frame. Call from scene.draw()."""
        if self._done:
            return

        t = self._elapsed

        if t < self._t_in:
            progress = _smoothstep(t / self._t_in)
            alpha = int(255 * progress)
            slide_y = int(self.SLIDE_PX * (1.0 - progress))
        elif t < self._t_in + self._t_hold:
            alpha = 255
            slide_y = 0
        else:
            progress = _smoothstep((t - self._t_in - self._t_hold) / self._t_out)
            alpha = int(255 * (1.0 - progress))
            slide_y = 0

        panel_y = 80 + slide_y   # appear near top of screen

        panel_copy = self._panel.copy()
        panel_copy.set_alpha(alpha)
        surface.blit(panel_copy, (self._panel_x, panel_y))

    @property
    def is_done(self) -> bool:
        return self._done
