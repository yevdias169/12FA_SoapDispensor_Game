# Minigame Hub — Integration Handoff

This repo is a pygame **minigame hub** (`master.py`) plus the individual games
under `Minigames/`. It is meant to be triggered by an external app (e.g. the
QR-scan / T&Cs web flow) after some event, run the games, and then hand control
back.

This document is for whoever integrates the hub with the web stack.

---

## 1. The integration seam: launch `master.py` as a subprocess

**Do not import the games into your app.** They must run in their own Python
environment (Python 3.12 conda env, because of mediapipe) and only need a
display + camera. The clean seam is to spawn `master.py` with that env's
interpreter when your event fires:

```python
import os, subprocess

GAME_PYTHON = "/home/ydclaw/miniforge3/envs/game/bin/python"
MASTER_PY   = "/home/ydclaw/12FA_SoapDispensor_Game/master.py"

result = subprocess.run(
    [GAME_PYTHON, MASTER_PY, "--all"],          # --all = kiosk mode (see §3)
    env={**os.environ, "DISPLAY": ":0"},        # show on the Pi's monitor
)
# This call BLOCKS until the hub exits, then your app continues.
# result.returncode == 0 on a clean finish.
```

- The full path to the conda env's `python` means **no `conda activate`** is needed.
- Your web stack can be anything (system Python, Node, …); it talks to the games
  only through this subprocess call.

---

## 2. It needs a real display (not headless)

The games are pygame / OpenCV **windows**. They render to the Pi's graphical
session, so the launching process must have `DISPLAY` set (`:0` is the Pi's
attached monitor) and the Pi must be logged into its desktop.

If your web server runs as a **systemd service / daemon**, it has no `DISPLAY`
by default and the launch fails with *"could not connect to display."* Pass
`DISPLAY=:0` in the subprocess `env` as shown above (and ensure the desktop
session is active).

---

## 3. Run modes

| Command | Behaviour |
|---|---|
| `python master.py` | Opens the menu; user clicks a game or "All Games". Returns to menu after each game. Exits only when the window is closed. |
| `python master.py --all` | **Kiosk mode.** Skips the menu, runs every game in order, then **exits** — returning control to your app. This is what you want for the event-triggered flow. |

Inside any game: an on-screen **SKIP** button (or **ESC** / **S**) advances to
the next game; closing the window exits the whole hub.

---

## 4. Camera coordination (important)

Only **one process can use the Pi camera at a time.** If your QR scanner uses
the camera, it **must fully release it before** launching the hub — otherwise
the games get a "camera busy" black screen.

The hub clears any stale `rpicam-vid` on startup, but it **cannot** evict a
`picamera2`/`libcamera`/browser process that is still holding the sensor. So the
sequence must be: **scan → release camera → launch `master.py`.**

The camera backend is **auto-detected** (the Pi has the `rpicam-vid` binary, dev
machines don't). Override if ever needed:
```bash
MINIGAME_CAMERA_BACKEND=opencv python master.py   # force USB webcam
MINIGAME_CAMERA_BACKEND=rpicam python master.py   # force Pi camera
```

---

## 5. Environment / dependencies on the Pi

The games run in the **`game` conda env** (miniforge, Python 3.12) at
`/home/ydclaw/miniforge3/envs/game`. Packages used across the games:

```
pygame  opencv (cv2)  numpy  mediapipe  scikit-learn  scikit-image
hsemotion-onnx        # Emotion Scanner
dlib  scipy           # Cat Tracing
```

If a game's dependency is missing it fails **gracefully** — its traceback prints
to the console and the hub returns to the menu; it does not crash the launcher.

System side: the Pi camera uses the `rpicam-vid` binary (libcamera stack), not
a V4L2 webcam. See `Minigames/pi_camera.py` for how frames are read.

---

## 6. Getting the repo onto the Pi

Everything the games need (code **+** model assets like `model.p`,
`hand_landmarker.task`, the dlib `.dat`) is committed, so a `git clone` or a zip
of this repo is self-contained. The big dlib landmark model (~95 MB) is included.

If you copy rather than clone, exclude `**/.venv` and `**/__pycache__` (build
cruft, already git-ignored).

---

## 7. Quick smoke test

On the Pi, with the desktop up:
```bash
cd ~/12FA_SoapDispensor_Game
/home/ydclaw/miniforge3/envs/game/bin/python master.py
```
Click a game. If a game bounces straight back to the menu, its traceback is in
the terminal — usually a missing dependency (see §5).
