# Rubik's Cube Verifier

Computer-vision system that verifies whether a Rubik's Cube is solved using a live camera feed.  A cube is "solved" when every face shows a single uniform colour and the six face colours are all distinct.

---

## Install

```bash
cd rubiks-vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Running

### Live single-face verification (default)
```bash
python main.py run
```
Hold one face inside the on-screen guide box.  After `STABLE_FRAMES` (default 8) consecutive agreeing frames the verdict appears.

### Calibration (recommended before first use)
```bash
python main.py calibrate
```
Shows each of the 6 standard colours in turn.  Press **SPACE** when the matching solved face is inside the guide box.  Saves reference colours to `calibration.json`.  Re-run `python main.py run` afterwards.

### Full 6-face scan
```bash
python main.py scan-cube
```
Guides you to present each face.  Press **SPACE** to accept each face (the single-face verdict must show SOLVED first).  Reports the overall cube verdict.

### Still-image test (no camera needed)
```bash
python main.py test-image --path /path/to/face.png
```
Runs the full pipeline on a static image and prints per-sticker verdicts to stdout.  Also opens an annotated display window.

### Keyboard controls (live windows)
| Key | Action |
|-----|--------|
| SPACE | Capture / confirm |
| c | Switch to calibration mode |
| q / ESC | Quit |

---

## Tests (offline — no camera needed)

```bash
pytest
```

Synthetic fixture images are generated automatically in `tests/fixtures/` on the first run.

---

## macOS → Raspberry Pi portability

The only change required is in `config.py`:

| Setup | `CAMERA_BACKEND` |
|-------|-----------------|
| macOS (built-in or USB webcam) | `"opencv"` (default) |
| Pi 5 + ribbon-cable Pi Camera Module 3 | `"picamera2"` |
| Pi 5 + USB webcam | `"opencv"` |

For the ribbon-cable camera, also install picamera2 on the Pi:
```bash
sudo apt install -y python3-picamera2
```
No other code changes are needed — the camera abstraction layer handles the rest.

---

## How calibration works

When you run `python main.py calibrate`, the system:
1. Prompts you to hold each of the 6 solved face colours inside the guide box.
2. Captures the centre sticker's median BGR value.
3. Converts it to CIE-LAB and saves both representations to `calibration.json`.

On subsequent runs the classifier loads these values instead of the built-in defaults.  This makes the system robust to your specific cube's paint colours and your ambient lighting.

If `calibration.json` is absent, a warning is printed and the system falls back to broad reference values defined in `config.py`.

---

## Why CIE-LAB + CIEDE2000?

RGB and HSV distances are **not** perceptually uniform.  A small numerical difference in HSV near white or grey can correspond to a very noticeable colour change, while a large HSV difference near a saturated primary colour can be nearly invisible.

CIE-LAB maps colours to a space where equal numerical distances correspond roughly to equal perceived differences.  CIEDE2000 (ΔE₀₀) refines this with correction terms for lightness, chroma, and hue that make it even more perceptually accurate.

The practical benefit: a single threshold (`DELTA_E_MAX = 20`) works reliably across all six cube colours and under varying lighting.  The HSV fallback (`CLASSIFIER = "hsv"`) is provided for comparison but requires per-hue tuning.

---

## config.py reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_BACKEND` | `"opencv"` | `"opencv"` for macOS/USB; `"picamera2"` for Pi ribbon camera |
| `CAMERA_INDEX` | `0` | OpenCV device index (0 = first webcam) |
| `FRAME_WIDTH/HEIGHT` | `1280×720` | Requested capture resolution |
| `DETECTION_MODE` | `"guided"` | `"guided"` (fixed ROI) or `"auto"` (contour detection) |
| `ROI_FRAC` | `0.6` | Guided ROI size as a fraction of the shorter frame dimension |
| `WARP_SIZE` | `300` | Warped face image size in pixels (square) |
| `PATCH_FRAC` | `0.5` | Central fraction of each cell sampled per sticker |
| `CLASSIFIER` | `"lab"` | `"lab"` (CIEDE2000) or `"hsv"` (range fallback) |
| `DELTA_E_MAX` | `20.0` | Max ΔE₀₀ to accept a palette match; increase if faces are misclassified |
| `INTRA_FACE_DELTA_E` | `12.0` | Max spread among 9 stickers for a face to count as uniform; decrease for stricter |
| `STABLE_FRAMES` | `8` | Consecutive agreeing frames before emitting a verdict |
| `MAX_UNCERTAIN` | `1` | Glare-flagged stickers above this → RETRY verdict |
| `MIN_FACE_AREA_FRAC` | `0.1` | Auto-mode: minimum face contour area as fraction of frame area |
| `GLARE_V_THRESH` | `0.92` | V channel threshold above which a patch may be glare |
| `GLARE_S_THRESH` | `0.08` | S channel threshold below which a bright patch is flagged as glare |
| `CALIBRATION_PATH` | `"calibration.json"` | Path to the calibration file |

### Tuning tips
- **Faces constantly misclassified as unknown**: increase `DELTA_E_MAX` (e.g. to 30).
- **Two different colours both classify to the same label**: decrease `DELTA_E_MAX`.
- **Verdict flickers**: increase `STABLE_FRAMES`.
- **Auto mode misses the face**: decrease `MIN_FACE_AREA_FRAC` or switch to `"guided"`.
- **White face flagged as glare**: decrease `GLARE_V_THRESH` to 0.97 or increase `GLARE_S_THRESH`.

---

## Project structure

```
rubiks-vision/
├── config.py            All tunables — single source of truth
├── main.py              CLI: run | calibrate | scan-cube | test-image
├── src/
│   ├── camera.py        CameraSource ABC + OpenCVCamera + PiCamera2Camera
│   ├── face_detector.py Guided ROI + auto quad detection + perspective warp
│   ├── sticker_sampler.py 3×3 grid, central-patch median, glare detection
│   ├── color_classifier.py LAB/CIEDE2000 + HSV fallback
│   ├── calibrator.py    Interactive 6-colour capture → calibration.json
│   ├── verifier.py      Single-face + full-cube logic + temporal debouncer
│   └── visualizer.py    All on-screen drawing
└── tests/
    ├── fixtures/        Auto-generated synthetic PNGs
    ├── test_color_classifier.py
    ├── test_sticker_sampler.py
    └── test_verifier.py
```
