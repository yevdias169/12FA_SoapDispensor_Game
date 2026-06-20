"""
test_verifier.py — offline tests for single-face and full-cube verification.
No camera or physical cube needed.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import config
from src.sticker_sampler import StickerSample
from src.color_classifier import LabClassifier, Palette
from src.verifier import (
    check_face, check_cube, Debouncer,
    Verdict, FaceResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_palette() -> Palette:
    return Palette(config.DEFAULT_PALETTE)


def _make_classifier() -> LabClassifier:
    return LabClassifier(_make_palette(), delta_e_max=config.DELTA_E_MAX)


def _make_sample(
    index: int,
    bgr: tuple[int, int, int],
    uncertain: bool = False,
) -> StickerSample:
    return StickerSample(
        index=index,
        bgr=np.array(bgr, dtype=np.uint8),
        cell_rect=(0, 0, 100, 100),
        patch_rect=(25, 25, 50, 50),
        uncertain=uncertain,
    )


def _nine_samples(bgr: tuple[int, int, int]) -> list[StickerSample]:
    return [_make_sample(i, bgr) for i in range(9)]


# ---------------------------------------------------------------------------
# check_face tests
# ---------------------------------------------------------------------------

class TestCheckFace:
    def test_uniform_green_face_is_solved(self):
        clf = _make_classifier()
        samples = _nine_samples((0, 200, 0))  # green
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, config.MAX_UNCERTAIN)
        assert result.verdict == Verdict.SOLVED
        assert result.label == "green"

    def test_uniform_red_face_is_solved(self):
        clf = _make_classifier()
        samples = _nine_samples((0, 0, 200))  # red
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, config.MAX_UNCERTAIN)
        assert result.verdict == Verdict.SOLVED
        assert result.label == "red"

    def test_one_off_sticker_is_unsolved(self):
        clf = _make_classifier()
        # 8 green, 1 blue centre
        samples = [_make_sample(i, (0, 200, 0)) for i in range(9)]
        samples[4] = _make_sample(4, (200, 0, 0))  # blue sticker at centre
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, config.MAX_UNCERTAIN)
        assert result.verdict == Verdict.UNSOLVED
        assert 4 in result.offending_indices

    def test_offending_index_is_correct(self):
        clf = _make_classifier()
        samples = _nine_samples((0, 0, 200))  # all red
        samples[7] = _make_sample(7, (200, 0, 0))  # blue at index 7
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, config.MAX_UNCERTAIN)
        assert result.verdict == Verdict.UNSOLVED
        assert 7 in result.offending_indices
        assert 0 not in result.offending_indices

    def test_too_many_uncertain_returns_retry(self):
        clf = _make_classifier()
        # MAX_UNCERTAIN=1 means 2 uncertain → RETRY
        samples = [_make_sample(i, (0, 200, 0)) for i in range(9)]
        samples[0] = _make_sample(0, (0, 200, 0), uncertain=True)
        samples[1] = _make_sample(1, (0, 200, 0), uncertain=True)
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, max_uncertain=1)
        assert result.verdict == Verdict.RETRY

    def test_one_uncertain_allowed(self):
        clf = _make_classifier()
        samples = [_make_sample(i, (0, 200, 0)) for i in range(9)]
        samples[0] = _make_sample(0, (0, 200, 0), uncertain=True)
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, max_uncertain=1)
        # 1 uncertain ≤ max_uncertain → should still get a verdict
        assert result.verdict in (Verdict.SOLVED, Verdict.UNSOLVED)

    def test_very_mixed_face_is_unsolved(self):
        clf = _make_classifier()
        colours = [
            (255, 255, 255), (0, 255, 255), (0, 0, 200),
            (0, 128, 255),   (0, 200, 0),   (200, 0, 0),
            (255, 255, 255), (0, 255, 255), (0, 0, 200),
        ]
        samples = [_make_sample(i, c) for i, c in enumerate(colours)]
        result = check_face(samples, clf, config.INTRA_FACE_DELTA_E, config.MAX_UNCERTAIN)
        assert result.verdict == Verdict.UNSOLVED


# ---------------------------------------------------------------------------
# Debouncer tests
# ---------------------------------------------------------------------------

class TestDebouncer:
    def test_analyzing_before_stable_frames(self):
        d = Debouncer(stable_frames=5)
        fr = FaceResult(verdict=Verdict.SOLVED, label="green")
        for _ in range(4):
            out = d.update(fr)
        assert out.verdict == Verdict.ANALYZING

    def test_solved_after_stable_frames(self):
        d = Debouncer(stable_frames=5)
        fr = FaceResult(verdict=Verdict.SOLVED, label="green")
        out = None
        for _ in range(5):
            out = d.update(fr)
        assert out is not None and out.verdict == Verdict.SOLVED

    def test_mixed_frames_stay_analyzing(self):
        d = Debouncer(stable_frames=3)
        frames = [
            FaceResult(verdict=Verdict.SOLVED),
            FaceResult(verdict=Verdict.UNSOLVED),
            FaceResult(verdict=Verdict.SOLVED),
        ]
        out = None
        for fr in frames:
            out = d.update(fr)
        assert out is not None and out.verdict == Verdict.ANALYZING

    def test_reset_clears_history(self):
        d = Debouncer(stable_frames=3)
        fr = FaceResult(verdict=Verdict.SOLVED)
        for _ in range(3):
            d.update(fr)
        d.reset()
        out = d.update(fr)
        assert out.verdict == Verdict.ANALYZING

    def test_stability_counter(self):
        d = Debouncer(stable_frames=5)
        fr = FaceResult(verdict=Verdict.SOLVED)
        for i in range(1, 6):
            d.update(fr)
            assert d.stability == i


# ---------------------------------------------------------------------------
# check_cube tests
# ---------------------------------------------------------------------------

class TestCheckCube:
    def _solved_result(self, label: str) -> FaceResult:
        return FaceResult(verdict=Verdict.SOLVED, label=label)

    def test_six_distinct_solved_faces_is_solved(self):
        face_results = [
            self._solved_result("white"),
            self._solved_result("yellow"),
            self._solved_result("red"),
            self._solved_result("orange"),
            self._solved_result("green"),
            self._solved_result("blue"),
        ]
        result = check_cube(face_results)
        assert result.verdict == Verdict.SOLVED

    def test_duplicate_face_colour_is_unsolved(self):
        face_results = [
            self._solved_result("white"),
            self._solved_result("yellow"),
            self._solved_result("red"),
            self._solved_result("orange"),
            self._solved_result("green"),
            self._solved_result("green"),  # duplicate!
        ]
        result = check_cube(face_results)
        assert result.verdict == Verdict.UNSOLVED
        assert "green" in result.duplicate_colors

    def test_one_unsolved_face_fails_cube(self):
        face_results = [
            self._solved_result("white"),
            self._solved_result("yellow"),
            FaceResult(verdict=Verdict.UNSOLVED, label="red", offending_indices=[4]),
            self._solved_result("orange"),
            self._solved_result("green"),
            self._solved_result("blue"),
        ]
        result = check_cube(face_results)
        assert result.verdict == Verdict.UNSOLVED
        assert 2 in result.failed_faces

    def test_all_same_colour_fails_cube(self):
        face_results = [self._solved_result("white") for _ in range(6)]
        result = check_cube(face_results)
        assert result.verdict == Verdict.UNSOLVED

    def test_failed_face_indices_reported(self):
        face_results = [
            FaceResult(verdict=Verdict.UNSOLVED, label="white"),
            self._solved_result("yellow"),
            self._solved_result("red"),
            FaceResult(verdict=Verdict.UNSOLVED, label="orange"),
            self._solved_result("green"),
            self._solved_result("blue"),
        ]
        result = check_cube(face_results)
        assert 0 in result.failed_faces
        assert 3 in result.failed_faces
