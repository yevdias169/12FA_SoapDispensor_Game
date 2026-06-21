#!/usr/bin/env bash
#
# deploy.sh — push the local repo to the Raspberry Pi over rsync.
#
# Usage:   ./deploy.sh            (uses default host below)
#          PI_HOST=ydclaw@172.20.10.3 ./deploy.sh   (override host/IP)
#
# git does NOT sync to the Pi — this rsync is how Mac changes reach it.
# The camera backend is auto-detected at runtime (the Pi has `rpicam-vid`,
# dev machines don't), so no per-file config editing is needed here.

set -euo pipefail

PI_HOST="${PI_HOST:-ydclaw@claw-pi}"
LOCAL_DIR="/Users/yevin/12FA_SoapDispensor_Game/"
REMOTE_DIR="~/12FA_SoapDispensor_Game/"

echo "==> Syncing repo to ${PI_HOST} ..."
rsync -avz \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude '.git' \
  "${LOCAL_DIR}" "${PI_HOST}:${REMOTE_DIR}"

echo ""
echo "==> Done. On the Pi (conda env 'game'), from ~/12FA_SoapDispensor_Game:"
echo "      Master hub          : python master.py"
echo "      Master hub (kiosk)  : python master.py --all   # all games, then exit"
echo "   Or a single game from ~/12FA_SoapDispensor_Game/Minigames:"
echo "      Guessing game : python -m Guessing_Game.main"
echo "      RPS           : cd RPS-Game && python RPS_vs_Computer.py"
echo "      Rubik's       : cd rubiks-vision && python minigames/rubiks_checker.py"
