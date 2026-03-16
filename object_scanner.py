import cv2
import numpy as np
import time
import urllib.request
import os
import threading
import queue
import winsound  # Windows beep — no extra install needed

# ─────────────────────────────────────────
#  DOWNLOAD YOLOv3 MODEL FILES
# ─────────────────────────────────────────
FILES = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg":     "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "coco.names":     "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

for fname, url in FILES.items():
    if not os.path.exists(fname):
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, fname)
        print(f"  Done: {fname}")

# ─────────────────────────────────────────
#  LOAD YOLO
# ─────────────────────────────────────────
print("Loading YOLO model...")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names") as f:
    CLASSES = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

print(f"YOLO loaded! Knows {len(CLASSES)} object types.\n")

# ─────────────────────────────────────────
#  DANGER CATEGORIES
# ─────────────────────────────────────────
FIRE_OBJECTS = {"fire", "flame"}          # custom — YOLO doesn't detect fire natively
                                           # we'll use orange pixel detection for fire

SHARP_OBJECTS = {
    "knife", "scissors", "fork",
    "sword", "axe", "blade", "gun", "rifle", "pistol"
}

WARNING_OBJECTS = {
    "cat", "dog", "bear", "elephant",
    "horse", "cow", "sheep", "bird",
    "bottle", "wine glass", "cup"
}

# COCO classes that exist in the model
COCO_SHARP  = {"knife", "scissors", "fork"}
COCO_WARN   = {"cat", "dog", "bear", "horse", "cow", "sheep",
               "bird", "bottle", "wine glass", "cup", "cell phone"}

# Beep settings  (frequency Hz, duration ms)
BEEP_DANGER  = (1200, 300)   # high pitch — fire / sharp
BEEP_WARNING = (600,  150)   # lower — other flagged objects
BEEP_SCAN    = (880,  80)    # soft tick — normal object detected

beep_queue = queue.Queue()
last_beep = 0
BEEP_COOLDOWN = 1.2


def beep_worker():
    """Background thread so beeps don't block the video."""
    while True:
        freq, dur = beep_queue.get()
        try:
            winsound.Beep(freq, dur)
        except:
            pass
        beep_queue.task_done()


threading.Thread(target=beep_worker, daemon=True).start()


def trigger_beep(freq, dur):
    global last_beep
    now = time.time()
    if now - last_beep > BEEP_COOLDOWN:
        beep_queue.put((freq, dur))
        last_beep = now


# ─────────────────────────────────────────
#  FIRE DETECTION VIA COLOR
#  (YOLO doesn't detect fire — we use HSV orange/red pixel mask)
# ─────────────────────────────────────────
def detect_fire_regions(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Orange-red fire range
    lower1 = np.array([0,  120, 120])
    upper1 = np.array([20, 255, 255])
    lower2 = np.array([160, 120, 120])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Only flag if fire-colored region is large enough (avoid false positives)
    fire_pixels = cv2.countNonZero(mask)
    h, w = frame.shape[:2]
    fire_ratio = fire_pixels / (h * w)

    fire_boxes = []
    if fire_ratio > 0.04:   # more than 4% of frame is fire-colored
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                x, y, w2, h2 = cv2.boundingRect(cnt)
                fire_boxes.append((x, y, w2, h2))

    return fire_boxes


# ─────────────────────────────────────────
#  YOLO DETECTION
# ─────────────────────────────────────────
def detect_objects(frame, conf_threshold=0.45, nms_threshold=0.4):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box":        boxes[i],
                "confidence": confidences[i],
                "label":      CLASSES[class_ids[i]],
                "class_id":   class_ids[i],
            })
    return results


# ─────────────────────────────────────────
#  CLASSIFY DANGER LEVEL
# ─────────────────────────────────────────
def danger_level(label):
    if label in COCO_SHARP or label in SHARP_OBJECTS:
        return "DANGER"
    if label in COCO_WARN or label in WARNING_OBJECTS:
        return "WARNING"
    return "SAFE"


# ─────────────────────────────────────────
#  DRAW DETECTION BOX
# ─────────────────────────────────────────
COLORS = {
    "DANGER":  (0, 0, 255),      # red
    "WARNING": (0, 165, 255),    # orange
    "SAFE":    (0, 220, 80),     # green
    "FIRE":    (0, 80, 255),     # deep red-orange
}

def draw_box(frame, x, y, w, h, label, confidence, level):
    color = COLORS[level]
    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Label background
    text = f"{label.upper()}  {confidence*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Danger badge
    if level == "DANGER":
        badge = "! SHARP OBJECT !"
        cv2.putText(frame, badge, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif level == "WARNING":
        badge = "CAUTION"
        cv2.putText(frame, badge, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)


# ─────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────
detected_log = []   # rolling log of last 6 detections

def draw_hud(frame, detections, fire_detected, fps, frame_count):
    global detected_log
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    status = "FIRE DETECTED!" if fire_detected else f"Scanning... {len(detections)} object(s)"
    color  = (0, 80, 255) if fire_detected else (200, 200, 200)
    cv2.putText(frame, status, (w // 2 - 120, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "Press Q to quit", (w - 160, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Detection log (bottom right)
    for det in detections:
        entry = f"{det['label']} ({det['confidence']*100:.0f}%)"
        if entry not in [e[0] for e in detected_log]:
            detected_log.append((entry, danger_level(det['label']), time.time()))
    if fire_detected:
        detected_log.append(("FIRE", "FIRE", time.time()))

    # Keep only last 6 and last 4 seconds
    detected_log = [(e, l, t) for e, l, t in detected_log if time.time() - t < 4][-6:]

    if detected_log:
        overlay3 = frame.copy()
        box_h = len(detected_log) * 22 + 28
        cv2.rectangle(overlay3, (w - 240, h - box_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay3, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "DETECTED:", (w - 230, h - box_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        for i, (entry, level, _) in enumerate(detected_log):
            c = COLORS.get(level, (200, 200, 200))
            cv2.putText(frame, f"  {entry}", (w - 230, h - box_h + 38 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)

    # Flashing red border when danger
    has_danger = fire_detected or any(danger_level(d["label"]) == "DANGER" for d in detections)
    if has_danger and int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    return frame


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time  = time.time()
    frame_count = 0
    # Run YOLO every N frames (every 3 = ~10fps detection on slow CPU)
    DETECT_EVERY = 3
    last_detections = []
    last_fire = False

    print("Object Safety Scanner started!")
    print("Point camera at objects — it will identify them.")
    print("Sharp objects and fire trigger beep alerts.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ── Run detection every N frames ──
        if frame_count % DETECT_EVERY == 0:
            last_detections = detect_objects(frame)
            fire_boxes = detect_fire_regions(frame)
            last_fire = len(fire_boxes) > 0

            # Draw fire boxes
            for (fx, fy, fw, fh) in fire_boxes:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), COLORS["FIRE"], 3)
                cv2.putText(frame, "! FIRE !", (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 3)

            # Beep logic
            if last_fire:
                trigger_beep(*BEEP_DANGER)
            else:
                for det in last_detections:
                    lvl = danger_level(det["label"])
                    if lvl == "DANGER":
                        trigger_beep(*BEEP_DANGER)
                        break
                    elif lvl == "WARNING":
                        trigger_beep(*BEEP_WARNING)
                        break
                    else:
                        trigger_beep(*BEEP_SCAN)
                        break

        # ── Always draw last detections ──
        for det in last_detections:
            x, y, w2, h2 = det["box"]
            draw_box(frame, x, y, w2, h2,
                     det["label"], det["confidence"],
                     danger_level(det["label"]))

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time + 1e-9)
        prev_time = now

        frame = draw_hud(frame, last_detections, last_fire, fps, frame_count)

        cv2.imshow("AI Safety Scanner  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Scanner closed.")


if __name__ == "__main__":
    main()
