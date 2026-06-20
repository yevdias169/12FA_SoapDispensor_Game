"""
test_fsm.py — proves auto-advance, clear-debounce, and duplicate handling (Goal 2).

Drives CubeCheckerFSM with scripted FaceResult sequences.
No camera, no display, no pygame needed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.verifier import (
    CubeCheckerFSM, FaceResult, FaceStatus, Verdict, Pipeline,
)

# Small values so tests are fast
STABLE        = 4
STABLE_CLEAR  = 3
EXPECTED      = ["white", "yellow", "red", "orange", "green", "blue"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(color: str) -> FaceResult:
    return FaceResult(
        verdict=Verdict.SOLVED,
        status=FaceStatus.UNIFORM,
        label=color,
        color_label=color,
    )

def _non_uniform() -> FaceResult:
    return FaceResult(verdict=Verdict.UNSOLVED, status=FaceStatus.NON_UNIFORM)

def _no_face() -> FaceResult:
    return FaceResult(verdict=Verdict.RETRY, status=FaceStatus.NO_FACE)

def _make_fsm() -> CubeCheckerFSM:
    return CubeCheckerFSM(
        stable_frames=STABLE,
        stable_clear_frames=STABLE_CLEAR,
        expected_colors=EXPECTED,
    )

def _collect_events(fsm: CubeCheckerFSM, result: FaceResult) -> list[str]:
    return fsm.feed(result)


# ---------------------------------------------------------------------------
# Auto-advance (Goal 2 core)
# ---------------------------------------------------------------------------

class TestAutoAdvance:
    def test_no_verify_before_stable_frames(self):
        fsm = _make_fsm()
        events = []
        for _ in range(STABLE - 1):
            events.extend(fsm.feed(_uniform("red")))
        assert fsm.state == "SCANNING"
        assert not any(e.startswith(CubeCheckerFSM.EV_FACE_VERIFIED) for e in events)

    def test_auto_verify_at_stable_frames(self):
        fsm = _make_fsm()
        events = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_uniform("red")))
        assert any(e == f"{CubeCheckerFSM.EV_FACE_VERIFIED}:red" for e in events)
        assert fsm.state == "FACE_VERIFIED"
        assert "red" in fsm.captured

    def test_non_uniform_interruption_resets(self):
        """
        Feed STABLE-1 UNIFORM:red, then one NON_UNIFORM, then STABLE UNIFORM:red.
        Verify must fire only at the very end (counter resets on interruption).
        """
        fsm = _make_fsm()
        events = []
        for _ in range(STABLE - 1):
            events.extend(fsm.feed(_uniform("red")))
        events.extend(fsm.feed(_non_uniform()))
        assert not any(e.startswith(CubeCheckerFSM.EV_FACE_VERIFIED) for e in events)

        events = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_uniform("red")))
        assert any(e == f"{CubeCheckerFSM.EV_FACE_VERIFIED}:red" for e in events)

    def test_no_face_then_uniform_resets(self):
        """NO_FACE → UNIFORM must restart the stability counter."""
        fsm = _make_fsm()
        events = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_no_face()))
        # Now switch to UNIFORM — counter must restart; not yet stable
        events = []
        for _ in range(STABLE - 1):
            events.extend(fsm.feed(_uniform("green")))
        assert not any(e.startswith(CubeCheckerFSM.EV_FACE_VERIFIED) for e in events)


# ---------------------------------------------------------------------------
# Force verify (SPACE override)
# ---------------------------------------------------------------------------

class TestForceVerify:
    def test_force_verify_uniform_face(self):
        fsm = _make_fsm()
        events = fsm.force_verify(_uniform("blue"))
        assert any(e == f"{CubeCheckerFSM.EV_FACE_VERIFIED}:blue" for e in events)
        assert fsm.state == "FACE_VERIFIED"

    def test_force_verify_non_uniform_returns_no_uniform(self):
        fsm = _make_fsm()
        events = fsm.force_verify(_non_uniform())
        assert CubeCheckerFSM.EV_NO_UNIFORM in events
        assert fsm.state == "SCANNING"

    def test_force_verify_ignored_in_face_verified_state(self):
        fsm = _make_fsm()
        fsm.force_verify(_uniform("blue"))   # enter FACE_VERIFIED
        assert fsm.state == "FACE_VERIFIED"
        events = fsm.force_verify(_uniform("red"))
        assert events == []                   # ignored in non-SCANNING state


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------

class TestDuplicateHandling:
    def test_duplicate_emits_event(self):
        fsm = _make_fsm()
        # Capture red
        for _ in range(STABLE):
            fsm.feed(_uniform("red"))
        # Advance back to SCANNING manually
        fsm.state = "SCANNING"
        fsm.tracker.reset()
        # Show red again — should emit DUPLICATE
        events = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_uniform("red")))
        assert any(e.startswith(CubeCheckerFSM.EV_DUPLICATE) for e in events)

    def test_duplicate_does_not_capture_again(self):
        fsm = _make_fsm()
        for _ in range(STABLE):
            fsm.feed(_uniform("red"))
        count_before = len(fsm.captured)
        fsm.state = "SCANNING"
        fsm.tracker.reset()
        for _ in range(STABLE):
            fsm.feed(_uniform("red"))
        assert len(fsm.captured) == count_before   # still only one "red"

    def test_different_color_is_not_duplicate(self):
        fsm = _make_fsm()
        for _ in range(STABLE):
            fsm.feed(_uniform("red"))
        fsm.state = "SCANNING"
        fsm.tracker.reset()
        events = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_uniform("blue")))
        assert any(e == f"{CubeCheckerFSM.EV_FACE_VERIFIED}:blue" for e in events)


# ---------------------------------------------------------------------------
# Clear-debounce
# ---------------------------------------------------------------------------

class TestClearDebounce:
    def _reach_face_verified(self, fsm: CubeCheckerFSM, color: str = "red") -> None:
        for _ in range(STABLE):
            fsm.feed(_uniform(color))
        assert fsm.state == "FACE_VERIFIED"

    def test_no_advance_before_clear_frames(self):
        fsm = _make_fsm()
        self._reach_face_verified(fsm)
        for _ in range(STABLE_CLEAR - 1):
            fsm.tick_clear_debounce(_no_face())
        # Toast is None (no pygame), so toast_done=True; but clear not ready yet
        assert not fsm.clear_ready

    def test_advance_after_clear_frames(self):
        fsm = _make_fsm()
        self._reach_face_verified(fsm)
        for _ in range(STABLE_CLEAR):
            fsm.tick_clear_debounce(_no_face())
        assert fsm.clear_ready
        events = fsm.advance_from_face_verified()
        assert CubeCheckerFSM.EV_BACK_SCANNING in events
        assert fsm.state == "SCANNING"

    def test_non_qualifying_frame_resets_clear_count(self):
        """NON_UNIFORM does not count toward the clear; clears consecutiveness."""
        fsm = _make_fsm()
        self._reach_face_verified(fsm)
        for _ in range(STABLE_CLEAR - 1):
            fsm.tick_clear_debounce(_no_face())
        fsm.tick_clear_debounce(_non_uniform())   # breaks streak
        assert not fsm.clear_ready
        assert fsm.clear_progress == 0

    def test_same_color_uniform_does_not_clear(self):
        """Showing the just-verified face again does NOT count as cleared."""
        fsm = _make_fsm()
        self._reach_face_verified(fsm, "red")
        for _ in range(STABLE_CLEAR):
            fsm.tick_clear_debounce(_uniform("red"))   # same color!
        assert not fsm.clear_ready

    def test_different_color_uniform_does_clear(self):
        """Showing a DIFFERENT uniform face counts toward the clear."""
        fsm = _make_fsm()
        self._reach_face_verified(fsm, "red")
        for _ in range(STABLE_CLEAR):
            fsm.tick_clear_debounce(_uniform("blue"))
        assert fsm.clear_ready


# ---------------------------------------------------------------------------
# Full 6-face sequence
# ---------------------------------------------------------------------------

class TestFullCubeScan:
    def _scan_color(self, fsm: CubeCheckerFSM, color: str) -> None:
        """Scan one face: reach FACE_VERIFIED, then clear-debounce back to SCANNING."""
        for _ in range(STABLE):
            fsm.feed(_uniform(color))
        assert fsm.state == "FACE_VERIFIED", f"Expected FACE_VERIFIED after {color}"
        for _ in range(STABLE_CLEAR):
            fsm.tick_clear_debounce(_no_face())
        events = fsm.advance_from_face_verified()
        # Either back to SCANNING or ALL_DONE
        assert any(
            e in (CubeCheckerFSM.EV_BACK_SCANNING, CubeCheckerFSM.EV_ALL_DONE)
            for e in events
        )

    def test_six_faces_reach_all_done(self):
        fsm = _make_fsm()
        for color in EXPECTED:
            self._scan_color(fsm, color)
        assert fsm.state == "ALL_DONE"
        assert len(fsm.captured) == 6

    def test_remaining_colors_tracked(self):
        fsm = _make_fsm()
        self._scan_color(fsm, "red")
        self._scan_color(fsm, "blue")
        remaining = fsm.remaining_colors
        assert "red"  not in remaining
        assert "blue" not in remaining
        assert len(remaining) == 4

    def test_reset_session_clears_everything(self):
        fsm = _make_fsm()
        self._scan_color(fsm, "red")
        fsm.reset_session()
        assert fsm.state == "SCANNING"
        assert len(fsm.captured) == 0
        assert len(fsm.remaining_colors) == len(EXPECTED)

    def test_duplicate_face_across_session(self):
        """Two faces of same color: second must emit DUPLICATE, not ALL_DONE."""
        fsm = _make_fsm()
        self._scan_color(fsm, "red")
        # Restore SCANNING state (clear-debounce already did it)
        assert fsm.state == "SCANNING"
        events = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_uniform("red")))
        assert any(e.startswith(CubeCheckerFSM.EV_DUPLICATE) for e in events)
        assert len(fsm.captured) == 1   # still just "red"


# ---------------------------------------------------------------------------
# Face-7-of-6 regression  (fix: immediate EV_ALL_DONE + _final_face_pending)
# ---------------------------------------------------------------------------

class TestFaceSixCompletion:
    """
    Proves that:
    1. EV_ALL_DONE is emitted in the same event list as EV_FACE_VERIFIED for
       the 6th face — no extra clear-debounce cycle needed.
    2. FSM never transitions back to SCANNING after the 6th capture.
    3. The HUD face_num helper  min(n_captured, n_expected)  never exceeds
       total_faces at any point during a full scan.
    """

    def _scan_five_faces(self, fsm: CubeCheckerFSM) -> None:
        """Scan 5 of the 6 expected colours with full clear-debounce."""
        for color in EXPECTED[:5]:
            for _ in range(STABLE):
                fsm.feed(_uniform(color))
            assert fsm.state == "FACE_VERIFIED", f"stuck after {color}"
            for _ in range(STABLE_CLEAR):
                fsm.tick_clear_debounce(_no_face())
            fsm.advance_from_face_verified()
        assert len(fsm.captured) == 5
        assert fsm.state == "SCANNING"

    def test_sixth_capture_emits_all_done_immediately(self):
        """EV_ALL_DONE must appear in the same feed() batch as EV_FACE_VERIFIED."""
        fsm = _make_fsm()
        self._scan_five_faces(fsm)

        events: list[str] = []
        for _ in range(STABLE):
            events.extend(fsm.feed(_uniform(EXPECTED[5])))

        assert any(e == f"{CubeCheckerFSM.EV_FACE_VERIFIED}:{EXPECTED[5]}" for e in events)
        assert CubeCheckerFSM.EV_ALL_DONE in events

    def test_state_never_returns_to_scanning_after_sixth(self):
        """advance_from_face_verified after the 6th face must yield ALL_DONE, not SCANNING."""
        fsm = _make_fsm()
        self._scan_five_faces(fsm)

        for _ in range(STABLE):
            fsm.feed(_uniform(EXPECTED[5]))

        assert fsm.state == "FACE_VERIFIED"   # capture happened, toast pending

        events = fsm.advance_from_face_verified()
        assert fsm.state == "ALL_DONE"
        assert CubeCheckerFSM.EV_ALL_DONE in events
        assert CubeCheckerFSM.EV_BACK_SCANNING not in events

    def test_feeding_after_sixth_capture_never_reaches_scanning(self):
        """feed() must be a no-op (no SCANNING transition) once in FACE_VERIFIED with all 6 done."""
        fsm = _make_fsm()
        self._scan_five_faces(fsm)

        for _ in range(STABLE):
            fsm.feed(_uniform(EXPECTED[5]))

        assert fsm.state == "FACE_VERIFIED"
        for _ in range(20):
            fsm.feed(_uniform("white"))   # would re-trigger SCANNING in the old code
            assert fsm.state != "SCANNING", "FSM unexpectedly re-entered SCANNING"

    def test_hud_face_num_never_exceeds_total_faces(self):
        """
        Simulate the clamping the scene applies:  min(n_captured, n_expected).
        Must stay <= total_faces at every frame throughout the full 6-face scan.
        """
        fsm = _make_fsm()
        n_expected = len(EXPECTED)

        for i, color in enumerate(EXPECTED):
            for _ in range(STABLE):
                fsm.feed(_uniform(color))
                face_num = min(len(fsm.captured), n_expected)
                assert face_num <= n_expected, (
                    f"face_num {face_num} > total {n_expected} while feeding {color}"
                )

            # Check immediately after capture is stored
            face_num = min(len(fsm.captured), n_expected)
            assert face_num <= n_expected

            if i < len(EXPECTED) - 1:
                for _ in range(STABLE_CLEAR):
                    fsm.tick_clear_debounce(_no_face())
                fsm.advance_from_face_verified()

        assert fsm.state in ("FACE_VERIFIED", "ALL_DONE")
        # After full scan: face_num == n_expected, so "Show face N of M" is suppressed
        face_num = min(len(fsm.captured), n_expected)
        assert face_num == n_expected   # equals, not greater-than

    def test_hud_prompt_suppressed_when_all_done(self):
        """
        HUD guard:  face_num < total_faces  must be False once all 6 are captured.
        Reproduces the "face 7 of 6" display bug via the predicate alone.
        """
        fsm = _make_fsm()
        self._scan_five_faces(fsm)

        for _ in range(STABLE):
            fsm.feed(_uniform(EXPECTED[5]))

        n_expected = len(EXPECTED)
        face_num   = min(len(fsm.captured), n_expected)
        # The HUD uses  `if face_num < total_faces`  to decide whether to draw
        # the "Show face N of M" prompt.  With the fix it must be False here.
        assert not (face_num < n_expected), (
            f"HUD would still show prompt: face_num={face_num}, total={n_expected}"
        )
