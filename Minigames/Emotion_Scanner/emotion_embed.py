#! /usr/bin/env python3

"""Embeddable version of emotion_game.py (mediapipe-free, cv2 + hsemotion).

Same contract as flappybird_embed.FlappyBirdGame: the host owns pygame.init(),
the display surface, the clock, and the webcam (cv2.VideoCapture); this class
never opens/closes either and only draws into the surface it's given.
"""

import random

import cv2
import numpy as np
import pygame
from pygame.locals import KEYUP, K_ESCAPE

HOLD_SECONDS = 1.5
TARGET_EMOTIONS = ["Happiness", "Surprise", "Anger", "Sadness"]


def cv2_frame_to_surface(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.rot90(rgb)
    return pygame.surfarray.make_surface(rgb)


class EmotionGame:
    """One playthrough of the emotion-scanner minigame.

    Arguments:
    display_surface: Surface to draw into; scaled to fit whatever size it is.
    cap: An already-open cv2.VideoCapture, owned by the host (shared with
        other camera-based minigames so the device isn't repeatedly
        opened/closed).
    face_cascade: A cv2.CascadeClassifier, loaded once by the host.
    recognizer: An HSEmotionRecognizer, loaded once by the host (loading the
        ONNX model per round would be slow).
    """

    def __init__(self, display_surface, cap, face_cascade, recognizer):
        self.display_surface = display_surface
        self.cap = cap
        self.face_cascade = face_cascade
        self.recognizer = recognizer

        self.font_big = pygame.font.SysFont(None, 64)
        self.font_small = pygame.font.SysFont(None, 32)

        self.sequence = random.sample(TARGET_EMOTIONS, len(TARGET_EMOTIONS))
        self.current_index = 0
        self.hold_progress = 0.0
        self.detected_emotion = ""
        self.finished = False
        self.done = False
        self.quit_requested = False
        self._frame = None

    def handle_event(self, event):
        if event.type == KEYUP and event.key == K_ESCAPE:
            self.quit_requested = True
        elif event.type == KEYUP and self.finished:
            # Any key once the sequence is complete returns to the menu.
            self.done = True

    def update(self, dt):
        """Advance by one frame. dt is the seconds elapsed (host's clock)."""
        ret, frame = self.cap.read()
        if not ret:
            return
        self._frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        self.detected_emotion = ""
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_img = frame[y:y + h, x:x + w]
            self.detected_emotion, _ = self.recognizer.predict_emotions(
                face_img, logits=False)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not self.finished:
            target = self.sequence[self.current_index]
            if self.detected_emotion == target:
                self.hold_progress += dt
                if self.hold_progress >= HOLD_SECONDS:
                    self.current_index += 1
                    self.hold_progress = 0.0
                    if self.current_index >= len(self.sequence):
                        self.finished = True
            else:
                self.hold_progress = max(0.0, self.hold_progress - dt * 2)

    def draw(self):
        if self._frame is None:
            return
        size = self.display_surface.get_size()
        frame_surface = cv2_frame_to_surface(self._frame)
        frame_surface = pygame.transform.scale(frame_surface, size)
        self.display_surface.blit(frame_surface, (0, 0))

        overlay = pygame.Surface((size[0], 110), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.display_surface.blit(overlay, (0, 0))

        if not self.finished:
            target = self.sequence[self.current_index]
            progress_text = (
                f"Show: {target}  ({self.current_index + 1}/{len(self.sequence)})")
            text_surf = self.font_big.render(progress_text, True, (255, 255, 255))
            self.display_surface.blit(text_surf, (20, 15))

            detected_text = f"Detected: {self.detected_emotion or '...'}"
            detected_surf = self.font_small.render(detected_text, True, (200, 200, 200))
            self.display_surface.blit(detected_surf, (20, 75))

            bar_width = int(300 * (self.hold_progress / HOLD_SECONDS))
            pygame.draw.rect(self.display_surface, (60, 60, 60),
                              (size[0] - 320, 35, 300, 30))
            pygame.draw.rect(self.display_surface, (0, 220, 0),
                              (size[0] - 320, 35, bar_width, 30))
        else:
            msg = "You showed them all! Press any key to return to menu."
            win_surf = self.font_big.render(msg, True, (0, 255, 0))
            self.display_surface.blit(
                win_surf, (size[0] // 2 - win_surf.get_width() // 2, 35))
