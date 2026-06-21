#!/usr/bin/env bash
#
# deploy.sh — push the repo to the Raspberry Pi and switch all three minigames
# to the Pi Camera Module (rpicam backend) in one step.
#
# Usage:   ./deploy.sh            (uses default host below)
#          PI_HOST=ydclaw@172.20.10.3 ./deploy.sh   (override host/IP)
#
# git does NOT sync to the Pi — this rsync is how Mac changes reach it.

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

echo "==> Switching all minigames + master to the Pi camera (rpicam) ..."
# The sed address (/CAMERA_BACKEND/) ensures only the backend assignment line is
# touched, and matches both `= "opencv"` and `: str = "opencv"` forms. Idempotent.
ssh "${PI_HOST}" '
  cd ~/12FA_SoapDispensor_Game &&
  sed -i "/CAMERA_BACKEND/ s/\"opencv\"/\"rpicam\"/" master.py &&
  cd Minigames &&
  sed -i "/CAMERA_BACKEND/ s/\"opencv\"/\"rpicam\"/" Guessing_Game/config.py &&
  sed -i "/CAMERA_BACKEND/ s/\"opencv\"/\"rpicam\"/" RPS-Game/config.py &&
  sed -i "/CAMERA_BACKEND/ s/\"opencv\"/\"rpicam\"/" rubiks-vision/config.py &&
  echo "   master.py:    " && grep CAMERA_BACKEND ~/12FA_SoapDispensor_Game/master.py | head -1 &&
  echo "   Guessing_Game:" && grep CAMERA_BACKEND Guessing_Game/config.py | head -1 &&
  echo "   RPS-Game:     " && grep CAMERA_BACKEND RPS-Game/config.py | head -1 &&
  echo "   rubiks-vision:" && grep CAMERA_BACKEND rubiks-vision/config.py | head -1
'

echo ""
echo "==> Done. On the Pi (conda env 'game'), from ~/12FA_SoapDispensor_Game:"
echo "      Master hub    : python master.py"
echo "   Or a single game from ~/12FA_SoapDispensor_Game/Minigames:"
echo "      Guessing game : python -m Guessing_Game.main"
echo "      RPS           : cd RPS-Game && python RPS_vs_Computer.py"
echo "      Rubik's       : cd rubiks-vision && python minigames/rubiks_checker.py"
