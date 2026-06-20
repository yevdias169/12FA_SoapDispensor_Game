#RPS vs Computer
# Single-hand Rock Paper Scissor game against the computer.
# Press SPACE to start a round: computer randomly picks R/P/S,
# a countdown plays, then your hand gesture at "SHOOT" is captured and judged.

import cv2
import numpy as np
import os
import random
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition


CHOICES = ['rock', 'paper', 'scissor']

# Map recognized hand gesture -> RPS choice
GESTURE_TO_CHOICE = {
    'fist': 'rock',
    'five': 'paper',
    'three': 'scissor',
    'yeah': 'scissor',
}

COUNTDOWN_SEQUENCE = ['3', '2', '1', 'SHOOT!']
COUNTDOWN_STEP_SEC = 0.8
RESULT_DISPLAY_SEC = 3.0
ICON_SIZE = 100 # Width/height in pixels of the rock/paper/scissor icon overlay


def judge(player, computer):
    if player == computer:
        return 'Tie'
    beats = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}
    if beats[player] == computer:
        return 'You win'
    return 'Computer wins'


def load_icon(path):
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError('Could not load icon image: %s' % path)
    icon = cv2.resize(icon, (ICON_SIZE, ICON_SIZE))
    if icon.shape[2] == 3:
        # No alpha channel: treat as fully opaque
        alpha = np.full((ICON_SIZE, ICON_SIZE, 1), 255, dtype=np.uint8)
        icon = np.concatenate([icon, alpha], axis=2)
    return icon


def overlay_icon(img, icon, x, y):
    # Paste icon (with alpha channel) onto img at top-left corner (x, y)
    img_height, img_width, _ = img.shape
    h, w = icon.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img_width), min(y + h, img_height)
    if x1 <= x0 or y1 <= y0:
        return img

    icon_crop = icon[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = icon_crop[:, :, 3:4] / 255.0
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + icon_crop[:, :, :3] * alpha

    return img


def draw_hand_skeleton(img, disp, param):
    # Draw only the hand skeleton (no class/gesture text labels)
    img_height, img_width, _ = img.shape
    for p in param:
        if p['class'] is None:
            continue
        for i in range(21):
            x = int(p['keypt'][i, 0])
            y = int(p['keypt'][i, 1])
            if x > 0 and y > 0 and x < img_width and y < img_height:
                start = p['keypt'][disp.ktree[i], :]
                x_ = int(start[0])
                y_ = int(start[1])
                if x_ > 0 and y_ > 0 and x_ < img_width and y_ < img_height:
                    cv2.line(img, (x_, y_), (x, y), disp.color[i], 2)
                cv2.circle(img, (x, y), 5, disp.color[i], -1)
    return img


# Load mediapipe hand class (single hand only)
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)

# Load display class
disp = DisplayHand(max_num_hands=1)

# Start video capture
cap = cv2.VideoCapture(0)

# Load gesture recognition class
gest = GestureRecognition(mode='eval')

# Load rock/paper/scissor icons
icons = {
    'rock'   : load_icon(os.path.join(SCRIPT_DIR, 'images', 'Rockimage.png')),
    'paper'  : load_icon(os.path.join(SCRIPT_DIR, 'images', 'Paperimage.png')),
    'scissor': load_icon(os.path.join(SCRIPT_DIR, 'images', 'Scissorimage.png')),
}

# Game state machine: 'wait' -> 'countdown' -> 'result' -> 'wait'
state = 'wait'
computer_choice = None
player_choice = None
result_text = None
countdown_start = None
result_start = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)

    img.flags.writeable = False
    param = pipe.forward(img)
    for p in param:
        if p['class'] is not None:
            p['gesture'] = gest.eval(p['angle'])
    img.flags.writeable = True

    img = draw_hand_skeleton(img, disp, param)

    # Show the rock/paper/scissor icon matching the current live gesture
    # in place of the left/right hand text label
    p = param[0]
    if p['class'] is not None:
        live_choice = GESTURE_TO_CHOICE.get(p['gesture'])
        if live_choice is not None:
            x = int(p['keypt'][0, 0]) - 30
            y = int(p['keypt'][0, 1]) + 40
            img = overlay_icon(img, icons[live_choice], x, y)

    img_height, img_width, _ = img.shape

    if state == 'countdown':
        elapsed = time.time() - countdown_start
        idx = int(elapsed // COUNTDOWN_STEP_SEC)
        if idx < len(COUNTDOWN_SEQUENCE):
            text = COUNTDOWN_SEQUENCE[idx]
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            x = int((img_width - size[0]) / 2)
            cv2.putText(img, text, (x, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Time's up: capture player's gesture and judge the round
            p = param[0]
            player_choice = GESTURE_TO_CHOICE.get(p['gesture'])
            if player_choice is None:
                result_text = 'No hand gesture detected'
            else:
                result_text = judge(player_choice, computer_choice)
            state = 'result'
            result_start = time.time()

    elif state == 'result':
        line1 = 'You: %s' % (player_choice.upper() if player_choice else 'NONE')
        line2 = 'Computer: %s' % computer_choice.upper()
        line3 = result_text.upper()

        for i, line in enumerate([line1, line2, line3]):
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = int((img_width - size[0]) / 2)
            cv2.putText(img, line, (x, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if time.time() - result_start > RESULT_DISPLAY_SEC:
            state = 'wait'

    else: # state == 'wait'
        text = 'Press SPACE to play'
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x = int((img_width - size[0]) / 2)
        cv2.putText(img, text, (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Game: Rock Paper Scissor vs Computer', img)

    key = cv2.waitKey(1)
    if key == 27: # Esc
        break
    elif key == 32 and state == 'wait': # Space
        computer_choice = random.choice(CHOICES)
        state = 'countdown'
        countdown_start = time.time()

pipe.pipe.close()
cap.release()
cv2.destroyAllWindows()
