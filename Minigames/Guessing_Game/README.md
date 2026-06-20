# Finger Guessing Game

A minigame where the computer picks a random number 1–5 and you hold up that
many fingers. A MediaPipe + scikit-learn model classifies your hand signal.

## How it works

1. A random target (1–5) is displayed on screen.
2. Hold up that many fingers in front of the webcam.
3. A live prediction updates continuously so you can see what the model thinks.
4. Press **SPACE** to lock in your guess.
5. **Correct** (green) → press any key to quit. **Incorrect** (red) → press any
   key to try again with a new target, or **Q** to quit.

## Run directly

From the project root (`Hand Landmark Detection/`):

```bash
python -m Guessing_Game.main
```

## Integrate with the pygame master controller

```python
from Guessing_Game import run_game

# ... master setup ...
run_game()          # blocks until the player quits; returns cleanly
# ... master continues ...
```

`run_game()` does not call `sys.exit()`, so returning from it leaves the
parent process alive.

## macOS camera permission

On first run macOS will prompt for camera access. If it does not appear, go to:
**System Settings → Privacy & Security → Camera** and enable access for Terminal
(or your IDE).

## Configuration

Edit `config.py` to change:

| Setting | Default | Notes |
|---------|---------|-------|
| `CAMERA_INDEX` | `1` | Matches existing project scripts; try `0` for built-in FaceTime camera |
| `MODEL_PATH` | `../model.p` | Points to the trained weights in the project root |
| `HAND_LANDMARKER_PATH` | `../hand_landmarker.task` | MediaPipe task file in the project root |
| `WINDOW_WIDTH/HEIGHT` | `960 × 540` | |
| `FPS` | `30` | |

## File structure

```
Guessing_Game/
├── __init__.py        # exposes run_game()
├── config.py          # paths, camera index, window size, colours
├── model_wrapper.py   # ModelWrapper: loads model once, exposes predict(frame) -> int
├── game.py            # pygame game loop and run_game() entry point
├── main.py            # __main__ shim for python -m Guessing_Game.main
└── README.md
```

No files outside `Guessing_Game/` were created or modified.
The trained weights (`model.p`, `hand_landmarker.task`) are referenced
in place — they are not duplicated.
