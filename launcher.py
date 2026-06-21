#! /usr/bin/env python3
"""
launcher.py — Pi-side trigger that waits for the web flow to signal the
dispense event, then launches the minigame hub.

Run this on the Pi (in the background, or on boot). It polls the teammate's
web service; when "dispensed" becomes true it runs every game in order
(master.py --all) and then exits.

Requires `requests` in whatever Python runs THIS file:
    pip install requests
The games themselves run in their own conda env via GAME_PYTHON below, so this
launcher does NOT need to run in the `game` env.
"""

import os
import subprocess
import time

import requests

STATUS_URL   = "https://possibly-caboose-tint.ngrok-free.dev/status"
GAME_PYTHON  = "/home/ydclaw/miniforge3/envs/game/bin/python"
MASTER_PY    = "/home/ydclaw/12FA_SoapDispensor_Game/master.py"
POLL_SECONDS = 1


def main():
    print(f"Waiting for dispense event at {STATUS_URL} ...")
    while True:
        try:
            r = requests.get(STATUS_URL, timeout=5)
            dispensed = r.json().get("dispensed", False)
        except Exception as exc:
            # network blip / ngrok down / bad JSON — keep watching, don't die
            print(f"[launcher] poll failed ({exc}); retrying...")
            time.sleep(POLL_SECONDS)
            continue

        if dispensed:
            print("[launcher] dispense detected — launching minigames")
            subprocess.run(
                [GAME_PYTHON, MASTER_PY, "--all"],
                env={**os.environ, "DISPLAY": ":0"},   # show on the Pi's monitor
            )
            print("[launcher] minigames finished")
            break

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
