"""
verifier.py — single-face verdict, temporal debouncing, full-cube verification,
              per-frame pipeline, and the session FSM.

Public API for the pygame minigame
───────────────────────────────────
Pipeline.evaluate(frame) -> FaceResult
  Runs detect → sample → classify → interpret in one call.
  Returns a FaceResult whose .status is always one of:
    FaceStatus.NO_FACE      – no recognizable cube face visible
    FaceStatus.NON_UNIFORM  – face found, stickers disagree
    FaceStatus.UNIFORM      – face found, all stickers agree (.color_label is set)

StableTracker.update(result_key) -> (last_key, is_stable)
  reset-on-change counter (see §5 of the spec).  Fixes the re-trigger bug.

CubeCheckerFSM
  Testable state machine used by RubiksCheckerScene.  Feed FaceResult objects;
  call force_verify() for the SPACE override; call tick_clear_debounce() and
  advance_from_face_verified() from the scene while in FACE_VERIFIED.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.sticker_sampler import StickerSample


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Verdict(Enum):
    """Legacy verdict kept for backward-compat with existing tests."""
    SOLVED = auto()
    UNSOLVED = auto()
    RETRY = auto()
    ANALYZING = auto()


class FaceStatus(Enum):
    """Per-frame status returned by Pipeline.evaluate (no temporal buffering)."""
    NO_FACE = auto()       # blank / hand / unrecognised content
    NON_UNIFORM = auto()   # face detected, stickers disagree
    UNIFORM = auto()       # face detected, all stickers agree


# ---------------------------------------------------------------------------
# FaceResult  (backward-compatible; new fields have defaults)
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Result of evaluating a single face, for one frame."""
    # Legacy fields (existing tests use these)
    verdict: Verdict = Verdict.UNSOLVED
    label: str = ""
    offending_indices: list[int] = field(default_factory=list)
    offending_colors: list[str] = field(default_factory=list)
    intra_delta_e: float = 0.0
    uncertain_count: int = 0

    # New fields for the pygame pipeline
    status: FaceStatus = FaceStatus.NON_UNIFORM
    color_label: str = ""          # same as label; explicit alias for FSM code
    samples: list = field(default_factory=list)   # list[StickerSample]
    sticker_labels: list[str] = field(default_factory=list)  # per-sticker labels
    corners: Any = None            # np.ndarray | None (4×2 float32)
    fallback: bool = False         # True if auto-detect fell back to guide box


# ---------------------------------------------------------------------------
# check_face  (unchanged logic; sets new status field)
# ---------------------------------------------------------------------------

def check_face(
    samples: "list[StickerSample]",
    classifier,
    intra_face_delta_e: float,
    max_uncertain: int,
) -> FaceResult:
    """
    Decide SOLVED / UNSOLVED / RETRY for one face.
    Also sets the new .status field on the returned FaceResult.
    """
    certain = [s for s in samples if not s.uncertain]
    uncertain_count = len(samples) - len(certain)

    if uncertain_count > max_uncertain:
        return FaceResult(
            verdict=Verdict.RETRY,
            status=FaceStatus.NO_FACE,
            uncertain_count=uncertain_count,
        )

    labels_and_de: list[tuple[int, str, float]] = [
        (s.index, *classifier.classify(s.bgr)) for s in certain
    ]

    all_bgr = [s.bgr for s in samples]
    intra_de = classifier.max_pairwise_delta_e(all_bgr)

    unique_labels = {lbl for _, lbl, _ in labels_and_de if lbl != "unknown"}
    offending: list[tuple[int, str]] = []
    dominant_label = ""

    if len(unique_labels) == 1:
        dominant_label = next(iter(unique_labels))
        for idx, lbl, _ in labels_and_de:
            if lbl == "unknown":
                offending.append((idx, lbl))
    elif len(unique_labels) > 1:
        cnt = Counter(lbl for _, lbl, _ in labels_and_de if lbl != "unknown")
        dominant_label = cnt.most_common(1)[0][0] if cnt else ""
        for idx, lbl, _ in labels_and_de:
            if lbl != dominant_label:
                offending.append((idx, lbl))
    else:
        offending = [(idx, lbl) for idx, lbl, _ in labels_and_de]

    label_ok = len(offending) == 0
    tightness_ok = intra_de < intra_face_delta_e

    if label_ok and tightness_ok:
        return FaceResult(
            verdict=Verdict.SOLVED,
            status=FaceStatus.UNIFORM,
            label=dominant_label,
            color_label=dominant_label,
            intra_delta_e=intra_de,
            uncertain_count=uncertain_count,
        )

    # Determine per-frame status for UNSOLVED result
    if not dominant_label or dominant_label == "unknown":
        pf_status = FaceStatus.NO_FACE   # all-unknown → treat as blank
    else:
        pf_status = FaceStatus.NON_UNIFORM

    return FaceResult(
        verdict=Verdict.UNSOLVED,
        status=pf_status,
        label=dominant_label,
        color_label=dominant_label,
        offending_indices=[i for i, _ in offending],
        offending_colors=[c for _, c in offending],
        intra_delta_e=intra_de,
        uncertain_count=uncertain_count,
    )


# ---------------------------------------------------------------------------
# Pipeline  — runs the full per-frame pipeline; returns an annotated FaceResult
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Wraps FaceDetector + StickerSampler + classifier in a single evaluate() call.
    Every frame must call evaluate(); never skip frames to avoid the re-trigger bug.
    """

    def __init__(
        self,
        detector,    # FaceDetector
        sampler,     # StickerSampler
        classifier,  # LabClassifier | HsvClassifier
        intra_face_delta_e: float,
        max_uncertain: int,
    ) -> None:
        self.detector = detector
        self.sampler = sampler
        self.classifier = classifier
        self.intra_face_delta_e = intra_face_delta_e
        self.max_uncertain = max_uncertain

    def evaluate(self, frame: np.ndarray) -> FaceResult:
        """
        Run the full pipeline on one BGR frame.
        Always returns a FaceResult with .status set.
        Never raises; on detection failure falls back to guided ROI.
        """
        warped, corners, fallback = self.detector.detect(frame)
        samples = self.sampler.sample(warped)
        sticker_labels = [self.classifier.classify(s.bgr)[0] for s in samples]

        result = check_face(
            samples, self.classifier, self.intra_face_delta_e, self.max_uncertain
        )
        # Attach drawing data for the renderer
        result.samples = samples
        result.sticker_labels = sticker_labels
        result.corners = corners
        result.fallback = fallback
        return result

    @staticmethod
    def result_key(result: FaceResult) -> tuple:
        """Return a hashable key for StableTracker."""
        if result.status == FaceStatus.NO_FACE:
            return ("NO_FACE",)
        if result.status == FaceStatus.NON_UNIFORM:
            return ("NON_UNIFORM",)
        return ("UNIFORM", result.color_label)


# ---------------------------------------------------------------------------
# StableTracker  — reset-on-change counter (fixes the re-trigger bug)
# ---------------------------------------------------------------------------

class StableTracker:
    """
    Counts consecutive frames with the same result_key.
    Any change resets the counter immediately — never stale.

    This is the core fix for Goal 1: when the cube is misaligned the key
    changes (UNIFORM→NON_UNIFORM), the count resets, and re-aligning
    starts a fresh count rather than continuing a stale "UNSOLVED" latch.
    """

    def __init__(self, stable_frames: int) -> None:
        self.stable_frames = stable_frames
        self._last_key: tuple | None = None
        self._count: int = 0

    def update(self, result_key: tuple) -> tuple[tuple | None, bool]:
        """
        Feed the current frame's key.
        Returns (last_key, is_stable).
        """
        if result_key != self._last_key:
            self._last_key = result_key   # change → restart
            self._count = 1
        else:
            self._count += 1
        is_stable = self._count >= self.stable_frames
        return self._last_key, is_stable

    def reset(self) -> None:
        self._last_key = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Legacy Debouncer  (kept; tests use it)
# ---------------------------------------------------------------------------

class Debouncer:
    """Legacy ring-buffer debouncer.  Replaced by StableTracker in the scene."""

    def __init__(self, stable_frames: int) -> None:
        self.stable_frames = stable_frames
        self._history: deque[Verdict] = deque(maxlen=stable_frames)

    def update(self, result: FaceResult) -> FaceResult:
        self._history.append(result.verdict)
        if len(self._history) < self.stable_frames:
            return FaceResult(verdict=Verdict.ANALYZING)
        if all(v == Verdict.SOLVED for v in self._history):
            return result
        if all(v == Verdict.UNSOLVED for v in self._history):
            return result
        if all(v == Verdict.RETRY for v in self._history):
            return FaceResult(verdict=Verdict.RETRY)
        return FaceResult(verdict=Verdict.ANALYZING)

    def reset(self) -> None:
        self._history.clear()

    @property
    def stability(self) -> int:
        if not self._history:
            return 0
        last = self._history[-1]
        count = 0
        for v in reversed(self._history):
            if v == last:
                count += 1
            else:
                break
        return count


# ---------------------------------------------------------------------------
# CubeCheckerFSM  — testable session state machine
# ---------------------------------------------------------------------------

class CubeCheckerFSM:
    """
    Finite state machine for the six-face scanning session.

    Drive it by calling feed(face_result) each frame.
    The scene calls tick_clear_debounce(face_result) inside FACE_VERIFIED and
    advance_from_face_verified() once the toast finishes + clear-debounce passes.

    All pygame / camera / UI concerns live in RubiksCheckerScene; this class
    has zero pygame dependencies so it can be driven by tests.

    States: SCANNING → FACE_VERIFIED → SCANNING (repeat) → ALL_DONE
    """

    # Events emitted by feed() / force_verify()
    EV_FACE_VERIFIED = "FACE_VERIFIED"
    EV_DUPLICATE     = "DUPLICATE"
    EV_NO_UNIFORM    = "NO_UNIFORM_TO_VERIFY"
    EV_ALL_DONE      = "ALL_DONE"
    EV_BACK_SCANNING = "BACK_TO_SCANNING"

    def __init__(
        self,
        stable_frames: int,
        stable_clear_frames: int,
        expected_colors: list[str],
    ) -> None:
        self.stable_frames = stable_frames
        self.stable_clear_frames = stable_clear_frames
        self.expected_colors = list(expected_colors)
        self._reset_session()

    # ------------------------------------------------------------------
    # Public session control
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Reset to initial SCANNING state, clearing all captured faces."""
        self._reset_session()

    def force_verify(self, face_result: FaceResult) -> list[str]:
        """
        SPACE override: verify the current face immediately if UNIFORM.
        Only meaningful in SCANNING state; ignores other states.
        """
        if self.state != "SCANNING":
            return []
        if face_result.status != FaceStatus.UNIFORM:
            return [self.EV_NO_UNIFORM]
        return self._capture_and_enter_verified(face_result.color_label, face_result)

    # ------------------------------------------------------------------
    # Per-frame feeds
    # ------------------------------------------------------------------

    def feed(self, face_result: FaceResult) -> list[str]:
        """
        Feed the current frame's FaceResult.
        Returns a list of event strings (may be empty).
        Call this every frame regardless of state.
        """
        if self.state == "SCANNING":
            return self._tick_scanning(face_result)
        # FACE_VERIFIED ticking is separate (needs toast_done from the scene)
        return []

    def tick_clear_debounce(self, face_result: FaceResult) -> None:
        """
        Called every frame while in FACE_VERIFIED.
        Increments clear_count when the frame qualifies as "cleared".
        Resets clear_count when it does not (consecutive-frames rule).
        """
        if self.state != "FACE_VERIFIED":
            return
        key = Pipeline.result_key(face_result)
        is_clear = (
            key[0] == "NO_FACE"
            or (key[0] == "UNIFORM" and key[1] != self._last_verified_color)
        )
        if is_clear:
            self._clear_count += 1
        else:
            self._clear_count = 0   # must be *consecutive* clear frames

    def advance_from_face_verified(self) -> list[str]:
        """
        Called by the scene once the toast animation is done AND
        clear_count >= stable_clear_frames.
        Transitions FACE_VERIFIED → SCANNING or ALL_DONE.
        Uses set equality (not a count) so a spurious duplicate never
        tricks the completion check.
        """
        if self.state != "FACE_VERIFIED":
            return []
        if set(self.captured.keys()) == set(self.expected_colors):
            self.state = "ALL_DONE"
            return [self.EV_ALL_DONE]
        self.state = "SCANNING"
        self.tracker.reset()
        return [self.EV_BACK_SCANNING]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def remaining_colors(self) -> list[str]:
        return [c for c in self.expected_colors if c not in self.captured]

    @property
    def clear_ready(self) -> bool:
        return self._clear_count >= self.stable_clear_frames

    @property
    def clear_progress(self) -> int:
        return self._clear_count

    @property
    def duplicate_hint(self) -> str | None:
        return self._duplicate_hint

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_session(self) -> None:
        self.state: str = "SCANNING"
        self.captured: dict[str, FaceResult] = {}
        self.tracker = StableTracker(self.stable_frames)
        self._last_verified_color: str = ""
        self._clear_count: int = 0
        self._duplicate_hint: str | None = None

    def _tick_scanning(self, face_result: FaceResult) -> list[str]:
        key = Pipeline.result_key(face_result)
        _, is_stable = self.tracker.update(key)

        if key[0] == "UNIFORM":
            color = key[1]
            if color in self.captured:
                self._duplicate_hint = color
                return [f"{self.EV_DUPLICATE}:{color}"]
            self._duplicate_hint = None
            if is_stable:
                return self._capture_and_enter_verified(color, face_result)
        else:
            self._duplicate_hint = None

        return []

    def _capture_and_enter_verified(
        self, color: str, face_result: FaceResult
    ) -> list[str]:
        self.captured[color] = face_result
        self.state = "FACE_VERIFIED"
        self._last_verified_color = color
        self._clear_count = 0
        events = [f"{self.EV_FACE_VERIFIED}:{color}"]
        # Detect completion immediately at capture time using set equality.
        # Emitting EV_ALL_DONE here lets the scene skip the clear-debounce
        # for the final face so we never reach "face 7 of 6".
        if set(self.captured.keys()) == set(self.expected_colors):
            events.append(self.EV_ALL_DONE)
        return events


# ---------------------------------------------------------------------------
# Full-cube verification  (used in ALL_DONE / CUBE_VERIFIED)
# ---------------------------------------------------------------------------

@dataclass
class CubeResult:
    verdict: Verdict
    face_results: list[FaceResult] = field(default_factory=list)
    failed_faces: list[int] = field(default_factory=list)
    duplicate_colors: list[str] = field(default_factory=list)


def check_cube(face_results: list[FaceResult]) -> CubeResult:
    """Verify that all 6 individually-SOLVED faces have 6 distinct colors."""
    failed_faces = [
        i for i, fr in enumerate(face_results) if fr.verdict != Verdict.SOLVED
    ]
    if failed_faces:
        return CubeResult(
            verdict=Verdict.UNSOLVED,
            face_results=face_results,
            failed_faces=failed_faces,
        )
    labels = [fr.label for fr in face_results]
    counts = Counter(labels)
    duplicates = [lbl for lbl, cnt in counts.items() if cnt > 1]
    if duplicates:
        return CubeResult(
            verdict=Verdict.UNSOLVED,
            face_results=face_results,
            duplicate_colors=duplicates,
        )
    return CubeResult(verdict=Verdict.SOLVED, face_results=face_results)
