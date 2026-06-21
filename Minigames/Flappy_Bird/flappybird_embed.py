#! /usr/bin/env python3

"""Embeddable version of the velocity-physics Flappy Bird game.

flappybird_velocity.py owns its own pygame.init()/display/event loop, so it
can't be dropped into a master file that hosts multiple minigames. This
module wraps the same Bird/PipePair logic in a FlappyBirdGame class that:

- draws into a display_surface supplied by the caller (never creates one),
- advances exactly one frame per handle_event()/update()/draw() call instead
  of running its own blocking while loop,
- never calls pygame.init()/pygame.quit(), so the host process keeps control
  of the pygame lifecycle,
- reports completion via the `done` and `quit_requested` flags rather than
  printing and returning.
"""

import pygame
from pygame.locals import (KEYUP, MOUSEBUTTONUP, K_ESCAPE, K_PAUSE, K_p,
                            K_UP, K_RETURN, K_SPACE)

from flappybird_velocity import (Bird, PipePair, load_images,
                                  msec_to_frames, WIN_WIDTH, WIN_HEIGHT)
from collections import deque


class FlappyBirdGame:
    """A single playthrough of Flappy Bird, embeddable in a host loop.

    The host is responsible for the pygame lifecycle (init/quit), the
    display surface, and the clock. For each iteration of the host's loop
    it should call handle_event() for every pending event, then update(),
    then draw().

    Attributes:
    done: True once the bird has crashed and the round is over.
    quit_requested: True if the player asked to exit the game entirely
        (e.g. pressed Escape), as opposed to merely losing.
    score: The current score.
    """

    def __init__(self, display_surface, images=None):
        """Initialise a new round of Flappy Bird.

        Arguments:
        display_surface: The Surface to draw into. Must be at least
            (WIN_WIDTH, WIN_HEIGHT) pixels; this class never resizes or
            creates a display of its own.
        images: Optional pre-loaded images dict (see load_images()), so a
            host running several rounds doesn't have to reload image files
            each time.
        """
        self.display_surface = display_surface
        self.images = images if images is not None else load_images()
        self.font = pygame.font.SysFont(None, 32, bold=True)
        self.reset()

    def reset(self):
        """Reset to the start of a new round."""
        self.bird = Bird(50, int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2), 0.0,
                          (self.images['bird-wingup'],
                           self.images['bird-wingdown']))
        self.pipes = deque()
        self.frame_clock = 0
        self.score = 0
        self.paused = False
        self.done = False
        self.quit_requested = False

    def handle_event(self, event):
        """Feed one pygame event to the game.

        Arguments:
        event: A pygame event, as obtained from pygame.event.get().
        """
        if event.type == KEYUP and event.key == K_ESCAPE:
            self.quit_requested = True
        elif event.type == KEYUP and event.key in (K_PAUSE, K_p):
            self.paused = not self.paused
        elif event.type == MOUSEBUTTONUP or (
                event.type == KEYUP and
                event.key in (K_UP, K_RETURN, K_SPACE)):
            if not self.done:
                self.bird.jump()

    def update(self):
        """Advance the game by one frame.

        Does nothing if the round is over or paused.
        """
        if self.done or self.paused:
            return

        if not self.frame_clock % msec_to_frames(PipePair.ADD_INTERVAL):
            self.pipes.append(
                PipePair(self.images['pipe-end'], self.images['pipe-body']))

        while self.pipes and not self.pipes[0].visible:
            self.pipes.popleft()

        for p in self.pipes:
            p.update()
        self.bird.update()

        for p in self.pipes:
            if p.x + PipePair.WIDTH < self.bird.x and not p.score_counted:
                self.score += 1
                p.score_counted = True

        pipe_collision = any(p.collides_with(self.bird) for p in self.pipes)
        if (pipe_collision or 0 >= self.bird.y or
                self.bird.y >= WIN_HEIGHT - Bird.HEIGHT):
            self.done = True

        self.frame_clock += 1

    def draw(self):
        """Draw the current frame to the display surface."""
        for x in (0, WIN_WIDTH / 2):
            self.display_surface.blit(self.images['background'], (x, 0))

        for p in self.pipes:
            self.display_surface.blit(p.image, p.rect)

        self.display_surface.blit(self.bird.image, self.bird.rect)

        score_surface = self.font.render(str(self.score), True,
                                          (255, 255, 255))
        score_x = WIN_WIDTH / 2 - score_surface.get_width() / 2
        self.display_surface.blit(score_surface,
                                   (score_x, PipePair.PIECE_HEIGHT))
