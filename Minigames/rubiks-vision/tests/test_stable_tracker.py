"""
test_stable_tracker.py — proves the re-trigger bug fix (Goal 1).

The StableTracker must:
  • reset its counter on ANY key change (no stale latches)
  • report is_stable only after STABLE_FRAMES consecutive identical keys
  • recover immediately when the cube is realigned after mis-alignment

No camera or display needed.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.verifier import StableTracker

STABLE = 5   # small value to keep tests short


class TestStableTrackerBasics:
    def test_not_stable_before_threshold(self):
        t = StableTracker(stable_frames=STABLE)
        for i in range(STABLE - 1):
            _, is_stable = t.update(("UNIFORM", "red"))
            assert not is_stable, f"Should not be stable after {i+1} frames"

    def test_stable_exactly_at_threshold(self):
        t = StableTracker(stable_frames=STABLE)
        for _ in range(STABLE):
            key, is_stable = t.update(("UNIFORM", "red"))
        assert is_stable
        assert key == ("UNIFORM", "red")

    def test_remains_stable_after_threshold(self):
        t = StableTracker(stable_frames=STABLE)
        for _ in range(STABLE + 3):
            _, is_stable = t.update(("NO_FACE",))
        assert is_stable

    def test_count_exposed(self):
        t = StableTracker(stable_frames=STABLE)
        for i in range(1, STABLE + 1):
            t.update(("NON_UNIFORM",))
            assert t.count == i

    def test_reset_clears_state(self):
        t = StableTracker(stable_frames=STABLE)
        for _ in range(STABLE):
            t.update(("UNIFORM", "blue"))
        t.reset()
        _, is_stable = t.update(("UNIFORM", "blue"))
        assert not is_stable
        assert t.count == 1


class TestStableTrackerResetOnChange:
    """
    Core bug-fix tests: any key change must restart the counter from 1.
    This is what prevents the "UNSOLVED latch" bug.
    """

    def test_change_resets_counter(self):
        t = StableTracker(stable_frames=STABLE)
        # Build up 4 frames of NO_FACE …
        for _ in range(STABLE - 1):
            t.update(("NO_FACE",))
        # … then change to UNIFORM:red — counter must restart
        _, is_stable = t.update(("UNIFORM", "red"))
        assert not is_stable
        assert t.count == 1, f"Expected count=1 after key change, got {t.count}"

    def test_no_face_to_uniform_resets(self):
        """Switching NO_FACE → UNIFORM resets correctly (realign-recovers test)."""
        t = StableTracker(stable_frames=STABLE)
        for _ in range(STABLE):          # reach stable NO_FACE
            t.update(("NO_FACE",))
        _, is_stable = t.update(("UNIFORM", "green"))
        assert not is_stable             # reset, only 1 UNIFORM frame so far
        assert t.count == 1

    def test_mid_sequence_non_uniform_resets(self):
        """
        Canonical sequence from the spec:
        [NO_FACE]*5 + [UNIFORM:red]*3 + [NON_UNIFORM]*2 + [UNIFORM:red]*STABLE
        Stability is reached only at the very end.
        """
        t = StableTracker(stable_frames=STABLE)

        for _ in range(5):
            _, stable = t.update(("NO_FACE",))
        # After 5 NO_FACE: stable only if STABLE<=5
        if STABLE > 5:
            assert not stable

        for _ in range(3):
            _, stable = t.update(("UNIFORM", "red"))
        assert not stable               # 3 < STABLE

        for _ in range(2):
            _, stable = t.update(("NON_UNIFORM",))
        assert not stable               # 2 < STABLE; also restarted from 1

        # Now feed exactly STABLE frames of UNIFORM:red → should reach stable
        for i in range(STABLE):
            _, stable = t.update(("UNIFORM", "red"))
        assert stable, "Should be stable after STABLE consecutive UNIFORM:red"

    def test_alternating_keys_never_stable(self):
        t = StableTracker(stable_frames=STABLE)
        for i in range(STABLE * 3):
            key = ("UNIFORM", "red") if i % 2 == 0 else ("NON_UNIFORM",)
            _, is_stable = t.update(key)
            assert not is_stable

    def test_uniform_different_colors_never_stable(self):
        """Alternating between two UNIFORM keys must never reach stability."""
        t = StableTracker(stable_frames=STABLE)
        colors = ["red", "blue"]
        for i in range(STABLE * 4):
            _, is_stable = t.update(("UNIFORM", colors[i % 2]))
            assert not is_stable


class TestStableTrackerReturnValues:
    def test_returns_last_key(self):
        t = StableTracker(stable_frames=3)
        t.update(("NO_FACE",))
        key, _ = t.update(("UNIFORM", "yellow"))
        assert key == ("UNIFORM", "yellow")

    def test_initial_state(self):
        t = StableTracker(stable_frames=3)
        assert t.count == 0
        assert t._last_key is None
