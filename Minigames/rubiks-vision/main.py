"""
main.py — CLI entry point for the Rubik's Cube verifier.

Subcommands:
  run           Live single-face verification (default).
  calibrate     Capture 6 face colours → calibration.json.
  scan-cube     Full 6-face verification (press SPACE per face).
  test-image    Run pipeline on a still image; print verdict.

Keyboard controls in live windows:
  SPACE   capture / confirm
  c       switch to calibration
  q/ESC   quit
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import config
from src.camera import create_camera
from src.face_detector import FaceDetector
from src.sticker_sampler import StickerSampler
from src.color_classifier import load_palette, create_classifier
from src.verifier import check_face, Debouncer, Verdict, check_cube, FaceResult
from src.visualizer import (
    draw_guide, draw_stickers, draw_hud, compute_warp_mapping
)
from src.calibrator import run_calibration


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------

def _build_components():
    """Construct the detector, sampler, classifier from config."""
    detector = FaceDetector(
        mode=config.DETECTION_MODE,
        roi_frac=config.ROI_FRAC,
        warp_size=config.WARP_SIZE,
        min_face_area_frac=config.MIN_FACE_AREA_FRAC,
    )
    sampler = StickerSampler(
        warp_size=config.WARP_SIZE,
        patch_frac=config.PATCH_FRAC,
        glare_v_thresh=config.GLARE_V_THRESH,
        glare_s_thresh=config.GLARE_S_THRESH,
    )
    palette = load_palette(config.CALIBRATION_PATH, config.DEFAULT_PALETTE)
    classifier = create_classifier(
        config.CLASSIFIER, palette, config.DELTA_E_MAX, config.HSV_PALETTE
    )
    return detector, sampler, classifier


def _process_frame(
    frame: np.ndarray,
    detector: FaceDetector,
    sampler: StickerSampler,
    classifier,
) -> tuple[np.ndarray, list, list[str], FaceResult, np.ndarray, bool]:
    """Run the full pipeline on one frame; return everything needed to draw."""
    warped, corners, fallback = detector.detect(frame)
    samples = sampler.sample(warped)
    labels = [classifier.classify(s.bgr)[0] for s in samples]
    result = check_face(
        samples, classifier, config.INTRA_FACE_DELTA_E, config.MAX_UNCERTAIN
    )
    return warped, samples, labels, result, corners, fallback


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(_args) -> None:
    """Live single-face verification loop."""
    detector, sampler, classifier = _build_components()
    debouncer = Debouncer(config.STABLE_FRAMES)

    with create_camera(config) as cam:
        print("Live verification started. Press q or ESC to quit.")
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[run] Camera read failed — retrying…")
                continue

            warped, samples, labels, result, corners, fallback = _process_frame(
                frame, detector, sampler, classifier
            )
            stable_result = debouncer.update(result)

            display = frame.copy()
            roi_rect = detector.guide_rect(frame)
            draw_guide(display, roi_rect, corners, fallback)

            offset, sx, sy = compute_warp_mapping(corners, config.WARP_SIZE)
            offending = stable_result.offending_indices if stable_result else []
            draw_stickers(display, samples, labels, offset, sx, sy, offending)

            draw_hud(
                display, stable_result,
                debouncer.stability, config.STABLE_FRAMES,
                config.DETECTION_MODE,
            )

            cv2.imshow("Rubik's Cube Verifier", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                cv2.destroyAllWindows()
                run_calibration(cam, detector, sampler, config.CALIBRATION_PATH)
                # Reload classifier with new calibration
                palette = load_palette(config.CALIBRATION_PATH, config.DEFAULT_PALETTE)
                classifier = create_classifier(
                    config.CLASSIFIER, palette, config.DELTA_E_MAX, config.HSV_PALETTE
                )
                debouncer.reset()

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Subcommand: calibrate
# ---------------------------------------------------------------------------

def cmd_calibrate(_args) -> None:
    """Interactive calibration flow."""
    detector, sampler, _ = _build_components()
    with create_camera(config) as cam:
        run_calibration(cam, detector, sampler, config.CALIBRATION_PATH)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Subcommand: scan-cube
# ---------------------------------------------------------------------------

def cmd_scan_cube(_args) -> None:
    """Full 6-face scan.  Press SPACE to accept each face."""
    detector, sampler, classifier = _build_components()
    face_names = ["White", "Yellow", "Red", "Orange", "Green", "Blue"]
    face_results: list[FaceResult] = []
    face_idx = 0

    print("\n=== Full Cube Scan ===")
    print("Press SPACE to accept each face when the verdict shows SOLVED.")
    print("Press q/ESC to abort.\n")

    with create_camera(config) as cam:
        debouncer = Debouncer(config.STABLE_FRAMES)
        while face_idx < 6:
            ok, frame = cam.read()
            if not ok:
                continue

            warped, samples, labels, result, corners, fallback = _process_frame(
                frame, detector, sampler, classifier
            )
            stable = debouncer.update(result)

            display = frame.copy()
            roi_rect = detector.guide_rect(frame)
            draw_guide(display, roi_rect, corners, fallback)
            offset, sx, sy = compute_warp_mapping(corners, config.WARP_SIZE)
            draw_stickers(
                display, samples, labels, offset, sx, sy, stable.offending_indices
            )
            draw_hud(
                display, stable, debouncer.stability,
                config.STABLE_FRAMES, config.DETECTION_MODE
            )

            prompt = (
                f"Face {face_idx+1}/6 ({face_names[face_idx]}): "
                "SPACE to accept  |  q to abort"
            )
            h = frame.shape[0]
            cv2.putText(
                display, prompt, (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3
            )
            cv2.putText(
                display, prompt, (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1
            )

            cv2.imshow("Cube Scan", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("[scan-cube] Aborted.")
                cv2.destroyAllWindows()
                return

            if key == ord(" "):
                # Accept current stable result regardless of verdict
                face_results.append(stable)
                print(
                    f"  Face {face_idx+1} ({face_names[face_idx]}): "
                    f"{stable.verdict.name}  label={stable.label!r}"
                )
                face_idx += 1
                debouncer.reset()

    cv2.destroyAllWindows()

    if len(face_results) < 6:
        print("Scan incomplete.")
        return

    cube_result = check_cube(face_results)
    print("\n=== Cube Verdict ===")
    print(f"  {cube_result.verdict.name}")
    if cube_result.failed_faces:
        print(f"  Faces that failed: {[f+1 for f in cube_result.failed_faces]}")
    if cube_result.duplicate_colors:
        print(f"  Duplicate face colours: {cube_result.duplicate_colors}")
    if cube_result.verdict == Verdict.SOLVED:
        print("  The cube is SOLVED!")


# ---------------------------------------------------------------------------
# Subcommand: test-image
# ---------------------------------------------------------------------------

def cmd_test_image(args) -> None:
    """Run the pipeline on a still image (no camera needed)."""
    path: str = args.path
    frame = cv2.imread(path)
    if frame is None:
        print(f"[test-image] Cannot read image: {path!r}", file=sys.stderr)
        sys.exit(1)

    detector, sampler, classifier = _build_components()
    warped, samples, labels, result, corners, fallback = _process_frame(
        frame, detector, sampler, classifier
    )

    print(f"File:    {path}")
    print(f"Verdict: {result.verdict.name}")
    print(f"Label:   {result.label!r}")
    print(f"Intra ΔE: {result.intra_delta_e:.2f}")
    for s, lbl in zip(samples, labels):
        unc = " [uncertain]" if s.uncertain else ""
        bad = " ← MISMATCH" if s.index in result.offending_indices else ""
        print(f"  Sticker {s.index}: {lbl:10s}  BGR={s.bgr.tolist()}{unc}{bad}")

    # Optionally display annotated image
    display = frame.copy()
    roi_rect = detector.guide_rect(frame)
    draw_guide(display, roi_rect, corners, fallback)
    offset, sx, sy = compute_warp_mapping(corners, config.WARP_SIZE)
    draw_stickers(display, samples, labels, offset, sx, sy, result.offending_indices)
    cv2.imshow("test-image result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rubik's Cube solved-state verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Live single-face verification (default)")
    sub.add_parser("calibrate", help="Capture 6 face colours → calibration.json")
    sub.add_parser("scan-cube", help="Full 6-face scan")
    p_img = sub.add_parser("test-image", help="Run pipeline on a still image")
    p_img.add_argument("--path", required=True, help="Path to the image file")

    args = parser.parse_args()

    dispatch = {
        "run":        cmd_run,
        "calibrate":  cmd_calibrate,
        "scan-cube":  cmd_scan_cube,
        "test-image": cmd_test_image,
        None:         cmd_run,  # default with no subcommand
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
