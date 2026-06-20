"""
render.py — convert OpenCV BGR frames to pygame surfaces and draw the sticker
            grid overlay in screen space.
"""

from __future__ import annotations

import numpy as np
import pygame

from src.verifier import FaceResult, FaceStatus


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def cv2_to_surface(frame_bgr: np.ndarray) -> pygame.Surface:
    """
    Convert a BGR numpy array from OpenCV into a pygame Surface.
    Uses an RGB view (free channel-order flip via numpy stride trick) then
    frombuffer for zero-copy performance.
    """
    frame_bgr = np.ascontiguousarray(frame_bgr)
    # BGR → RGB: reverse the colour channels (view, no allocation)
    frame_rgb = frame_bgr[:, :, ::-1]
    frame_rgb = np.ascontiguousarray(frame_rgb)
    w, h = frame_rgb.shape[1], frame_rgb.shape[0]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")


def fit_frame(
    surface: pygame.Surface,
    screen_w: int,
    screen_h: int,
) -> tuple[pygame.Surface, int, int, float]:
    """
    Scale surface to fill screen_w × screen_h with letterboxing.

    Returns (scaled_surface, offset_x, offset_y, scale_factor).
    scale_factor converts original frame pixel coords → screen pixel coords.
    """
    sw, sh = surface.get_size()
    scale = min(screen_w / sw, screen_h / sh)
    new_w = int(sw * scale)
    new_h = int(sh * scale)
    scaled = pygame.transform.smoothscale(surface, (new_w, new_h))
    ox = (screen_w - new_w) // 2
    oy = (screen_h - new_h) // 2
    return scaled, ox, oy, scale


# ---------------------------------------------------------------------------
# Sticker overlay
# ---------------------------------------------------------------------------

# Colour map: palette label → RGB for drawing in pygame
_LABEL_RGB: dict[str, tuple[int, int, int]] = {
    "white":   (255, 255, 255),
    "yellow":  (255, 230, 0  ),
    "red":     (220, 0,   0  ),
    "orange":  (255, 140, 0  ),
    "green":   (0,   200, 0  ),
    "blue":    (0,   0,   220),
    "unknown": (80,  80,  80 ),
}


def _map_warped_to_screen(
    wx: float, wy: float,
    corners: np.ndarray,
    warp_size: int,
    screen_scale: float,
    screen_ox: int,
    screen_oy: int,
) -> tuple[int, int]:
    """Map a point in warped-face coords to screen (pygame) coords."""
    tl = corners[0]
    tr = corners[1]
    bl = corners[3]
    face_w = float(np.linalg.norm(tr - tl))
    face_h = float(np.linalg.norm(bl - tl))
    sx_warp = face_w / warp_size
    sy_warp = face_h / warp_size
    fx = tl[0] + wx * sx_warp
    fy = tl[1] + wy * sy_warp
    return (
        int(fx * screen_scale + screen_ox),
        int(fy * screen_scale + screen_oy),
    )


def draw_sticker_grid(
    surface: pygame.Surface,
    result: FaceResult,
    warp_size: int,
    screen_scale: float,
    screen_ox: int,
    screen_oy: int,
    font: pygame.font.Font,
) -> None:
    """
    Draw the 3×3 sticker bounding boxes on the pygame surface.
    Offending stickers are highlighted in red; uncertain ones show a '?'.
    """
    if result.corners is None or not result.samples:
        return

    corners = result.corners
    offending = set(result.offending_indices)

    for sample, label in zip(result.samples, result.sticker_labels):
        cx, cy, cw, ch = sample.cell_rect

        tl_s = _map_warped_to_screen(cx,      cy,      corners, warp_size, screen_scale, screen_ox, screen_oy)
        br_s = _map_warped_to_screen(cx + cw, cy + ch, corners, warp_size, screen_scale, screen_ox, screen_oy)

        fw = max(1, br_s[0] - tl_s[0])
        fh = max(1, br_s[1] - tl_s[1])
        rect = pygame.Rect(tl_s[0], tl_s[1], fw, fh)

        # Border colour: red for mismatch, dim white otherwise
        border = (220, 50, 50) if sample.index in offending else (180, 180, 180)
        pygame.draw.rect(surface, border, rect, 2)

        # Colour swatch in top-left corner
        b, g, r = int(sample.bgr[0]), int(sample.bgr[1]), int(sample.bgr[2])
        swatch_size = max(6, min(fw, fh) // 3)
        swatch_rect = pygame.Rect(tl_s[0] + 3, tl_s[1] + 3, swatch_size, swatch_size)
        pygame.draw.rect(surface, (r, g, b), swatch_rect)
        pygame.draw.rect(surface, (0, 0, 0), swatch_rect, 1)

        # Label (abbreviated)
        lbl_color = _LABEL_RGB.get(label, (180, 180, 180))
        txt = font.render(label[:3], True, (0, 0, 0))
        txt2 = font.render(label[:3], True, lbl_color)
        pos = (tl_s[0] + 3, br_s[1] - txt.get_height() - 2)
        surface.blit(txt,  (pos[0] + 1, pos[1] + 1))
        surface.blit(txt2, pos)

        if sample.uncertain:
            q = font.render("?", True, (255, 165, 0))
            surface.blit(q, (br_s[0] - q.get_width() - 2, tl_s[1] + 2))


def draw_guide_box(
    surface: pygame.Surface,
    guide_rect: tuple[int, int, int, int],
    screen_scale: float,
    screen_ox: int,
    screen_oy: int,
    color: tuple[int, int, int] = (0, 220, 220),
    thickness: int = 2,
) -> None:
    """Draw the ROI guide rectangle scaled to screen coords."""
    gx, gy, gw, gh = guide_rect
    sx = int(gx * screen_scale + screen_ox)
    sy = int(gy * screen_scale + screen_oy)
    sw = int(gw * screen_scale)
    sh = int(gh * screen_scale)
    pygame.draw.rect(surface, color, pygame.Rect(sx, sy, sw, sh), thickness)
