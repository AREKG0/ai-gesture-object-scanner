import cv2
import pyautogui
import numpy as np
import time
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ─────────────────────────────────────────
#  DOWNLOAD MODEL
# ─────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~9MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

# ─────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────
#  PYAUTOGUI SETUP
# ─────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 640, 480
COOLDOWN = 0.8
last_action_time = 0

# ─────────────────────────────────────────
#  FINGER STATE
# ─────────────────────────────────────────
def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    states = []
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        states.append(1)
    else:
        states.append(0)
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
            states.append(1)
        else:
            states.append(0)
    return states

# ─────────────────────────────────────────
#  GESTURE DETECTION
# ─────────────────────────────────────────
def detect_gesture(fingers, landmarks):
    thumb, index, middle, ring, pinky = fingers
    if index == 1 and middle == 1 and ring == 0 and pinky == 0 and thumb == 0:
        return "volume_up",  "Peace -> Volume Up"
    if thumb == 1 and index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return "screenshot", "Open Hand -> Screenshot"
    if thumb == 0 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "play_pause", "Fist -> Play/Pause"
    if pinky == 1 and index == 0 and middle == 0 and ring == 0 and thumb == 0:
        return "next",       "Pinky -> Next"
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "prev",       "Thumb -> Previous"
    dist = np.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y)
    if dist < 0.05:
        return "click",      "Pinch -> Click"
    return None, None

# ─────────────────────────────────────────
#  PERFORM ACTION
# ─────────────────────────────────────────
def perform_action(action):
    global last_action_time
    now = time.time()
    if now - last_action_time < COOLDOWN:
        return False
    if action == "volume_up":    pyautogui.press("volumeup")
    elif action == "volume_down": pyautogui.press("volumedown")
    elif action == "screenshot":  pyautogui.hotkey("win", "prtsc")
    elif action == "play_pause":  pyautogui.press("playpause")
    elif action == "next":        pyautogui.press("right")
    elif action == "prev":        pyautogui.press("left")
    elif action == "click":
        pyautogui.click()
        return True
    last_action_time = now
    return True

# ─────────────────────────────────────────
#  MOUSE CONTROL
# ─────────────────────────────────────────
def move_mouse(fingers, landmarks):
    thumb, index, middle, ring, pinky = fingers
    if index == 1 and middle == 0 and ring == 0 and pinky == 0 and thumb == 0:
        sx = np.interp(landmarks[8].x, [0.1, 0.9], [0, SCREEN_W])
        sy = np.interp(landmarks[8].y, [0.1, 0.9], [0, SCREEN_H])
        pyautogui.moveTo(int(sx), int(sy), duration=0)
        return True
    return False

# ─────────────────────────────────────────
#  DRAW HAND SKELETON
# ─────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 180), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

# ─────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────
def draw_hud(frame, gesture_label, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    if gesture_label:
        cv2.putText(frame, gesture_label, (w // 2 - 160, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    legend = [
        "GESTURES:",
        "1 finger  = Move mouse",
        "2 fingers = Vol Up",
        "Fist      = Play/Pause",
        "Open hand = Screenshot",
        "Thumb     = Prev",
        "Pinky     = Next",
        "Pinch     = Click",
    ]
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - len(legend)*22 - 14), (220, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(legend):
        color = (0, 220, 255) if i == 0 else (200, 200, 200)
        cv2.putText(frame, line, (8, h - len(legend)*22 - 4 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    return frame

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    prev_time = time.time()
    active_label = None
    label_timer = 0

    print("\nGesture Controller started! Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand(frame, hand_landmarks, w, h)
                fingers = fingers_up(hand_landmarks)
                moving = move_mouse(fingers, hand_landmarks)
                if not moving:
                    action, label = detect_gesture(fingers, hand_landmarks)
                    if action:
                        if perform_action(action):
                            active_label = label
                            label_timer = time.time()

        gesture_label = active_label if (active_label and time.time() - label_timer < 1.0) else None
        if not gesture_label:
            active_label = None

        now = time.time()
        fps = 1 / (now - prev_time + 1e-9)
        prev_time = now

        frame = draw_hud(frame, gesture_label, fps)
        cv2.imshow("Gesture Controller  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Gesture Controller closed.")

if __name__ == "__main__":
    main()