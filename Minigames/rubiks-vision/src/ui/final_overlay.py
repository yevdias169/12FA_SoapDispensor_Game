"""
final_overlay.py — "CUBE SOLVE VERIFIED" celebration overlay.

Timeline:
  0 … FINAL_BACKDROP_FADE  : dark backdrop fades in (alpha 0 → 200)
  FINAL_BACKDROP_FADE … +FINAL_POP_TIME : text pops in (scale 0.6→1.0) + fades in
  Holds until FINAL_AUTO_DISMISS seconds total elapsed, or any key press.

Confetti: a few dozen gravity-driven coloured rects in the 6 cube colours.
"""

from __future__ import annotations

import random

import pygame


_CONFETTI_COLORS_RGB: list[tuple[int, int, int]] = [
    (255, 255, 255),  # white
    (255, 215, 0  ),  # yellow
    (210, 30,  30 ),  # red
    (255, 130, 0  ),  # orange
    (30,  180, 30 ),  # green
    (30,  80,  210),  # blue
]

_GRAVITY = 380.0   # px / s²


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "size", "life", "age")

    def __init__(
        self,
        x: float, y: float,
        vx: float, vy: float,
        color: tuple[int, int, int],
        size: int,
        life: float,
    ) -> None:
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.color = color
        self.size = size
        self.life = life
        self.age = 0.0

    def update(self, dt: float) -> None:
        self.vy += _GRAVITY * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.age += dt

    @property
    def alive(self) -> bool:
        return self.age < self.life

    def draw(self, surface: pygame.Surface) -> None:
        alpha = max(0, int(255 * (1.0 - self.age / self.life)))
        s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        s.fill((*self.color, alpha))
        surface.blit(s, (int(self.x), int(self.y)))


class FinalOverlay:
    """
    Full-screen celebration overlay shown in CUBE_VERIFIED state.
    Call draw(surface, dt) every frame; dismiss() on any key or is_done.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        backdrop_fade: float,
        pop_time: float,
        auto_dismiss: float,
    ) -> None:
        self._sw = screen_w
        self._sh = screen_h
        self._backdrop_fade = backdrop_fade
        self._pop_time = pop_time
        self._auto_dismiss = auto_dismiss
        self._elapsed = 0.0
        self._done = False

        # Fonts (created once)
        self._font_title = pygame.font.SysFont("Arial", 56, bold=True)
        self._font_check = pygame.font.SysFont("Arial", 72, bold=True)
        self._font_sub   = pygame.font.SysFont("Arial", 22)

        self._title_surf = self._font_title.render("CUBE SOLVED", True, (255, 255, 255))
        self._check_surf = self._font_check.render("✓", True, (0, 220, 80))
        self._sub_surf   = self._font_sub.render(
            "Press any key to continue", True, (160, 160, 160)
        )

        # Backdrop surface (static; alpha applied per-frame)
        self._backdrop = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        self._backdrop.fill((0, 0, 0, 200))

        # Spawn confetti
        self._particles: list[_Particle] = []
        for _ in range(60):
            self._particles.append(
                _Particle(
                    x=random.uniform(0, screen_w),
                    y=random.uniform(-screen_h * 0.3, 0),
                    vx=random.uniform(-80, 80),
                    vy=random.uniform(-220, -80),
                    color=random.choice(_CONFETTI_COLORS_RGB),
                    size=random.randint(6, 14),
                    life=random.uniform(2.0, 4.0),
                )
            )

    def update(self, dt: float) -> None:
        """Advance clocks and physics. Call from scene.update(dt)."""
        if self._done:
            return
        self._elapsed += dt
        for p in self._particles:
            if p.alive:
                p.update(dt)
        if self._elapsed >= self._auto_dismiss:
            self._done = True

    def draw(self, surface: pygame.Surface) -> None:
        """Render the current frame. Call from scene.draw()."""
        if self._done:
            return

        # Confetti (behind backdrop)
        for p in self._particles:
            if p.alive:
                p.draw(surface)

        # Backdrop fade
        bd_progress = _smoothstep(min(self._elapsed / self._backdrop_fade, 1.0))
        bd = self._backdrop.copy()
        bd.set_alpha(int(200 * bd_progress))
        surface.blit(bd, (0, 0))

        # Text pop-in after backdrop
        pop_start = self._backdrop_fade
        pop_elapsed = self._elapsed - pop_start
        if pop_elapsed >= 0:
            t = _smoothstep(min(pop_elapsed / self._pop_time, 1.0))
            scale = 0.6 + 0.4 * t
            text_alpha = int(255 * t)

            tw, th = self._title_surf.get_size()
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))
            scaled_title = pygame.transform.smoothscale(
                self._title_surf, (scaled_w, scaled_h)
            )
            scaled_title.set_alpha(text_alpha)

            cx = (self._sw - scaled_w) // 2
            cy = self._sh // 2 - 40

            cw, ch_h = self._check_surf.get_size()
            check_copy = self._check_surf.copy()
            check_copy.set_alpha(text_alpha)
            surface.blit(check_copy, ((self._sw - cw) // 2, cy - ch_h - 10))
            surface.blit(scaled_title, (cx, cy))

            if t >= 1.0:
                sub = self._sub_surf.copy()
                sub_alpha = int(255 * min(
                    (self._elapsed - pop_start - self._pop_time) / 0.5, 1.0
                ))
                sub.set_alpha(max(0, sub_alpha))
                surface.blit(
                    sub,
                    ((self._sw - sub.get_width()) // 2, cy + scaled_h + 16),
                )

    def dismiss(self) -> None:
        """Call on any key press to finish early."""
        self._done = True

    @property
    def is_done(self) -> bool:
        return self._done
