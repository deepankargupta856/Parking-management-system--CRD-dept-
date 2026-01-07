# corrected_parking.py

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import deque

from shared_state import slot_state  # 🔑 shared state (FIXED)

# =========================
# CONFIG
# =========================
TARGET_DIM = (1024, 768)
SMOOTH_WINDOW = 3
OCCUPANCY_REFRESH_INTERVAL = 5.0

# =========================
# GLOBALS
# =========================
current_polygon = []
polygons = []
slot_history = []
api_started = False

# =========================
# MOUSE CALLBACK
# =========================
def mouse_callback(event, x, y, flags, param):
    global current_polygon, polygons

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            polygons.append(np.array(current_polygon, dtype=np.int32))
        current_polygon.clear()

# =========================
# DRAWING HELPERS
# =========================
def draw_polygons(frame):
    # Draw finalized slots
    for idx, poly in enumerate(polygons):
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
        M = cv2.moments(poly)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, f"Slot {idx + 1}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw currently-being-drawn polygon
    if len(current_polygon) > 1:
        pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)

def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

# =========================
# MAIN
# =========================
def main(video_path="resources/AED Parking Video.mp4"):
    global slot_history, api_started

    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video source")
        return

    # ---- SLOT DEFINITION PHASE ----
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading first frame")
        return

    first_frame = cv2.resize(first_frame, TARGET_DIM)

    cv2.namedWindow("Define Parking Slots", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Define Parking Slots", mouse_callback)

    while True:
        temp = first_frame.copy()
        draw_polygons(temp)

        cv2.putText(
            temp,
            "Left click: add point | Right click: close slot | 's': start | 'q': quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Define Parking Slots", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # finalize last polygon if still open
            if len(current_polygon) >= 3:
                polygons.append(np.array(current_polygon, dtype=np.int32))
                current_polygon.clear()

            # start API AFTER slots are fixed
            if not api_started:
                from parking_api import start_api
                threading.Thread(target=start_api, daemon=True).start()
                api_started = True

            break

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Define Parking Slots")

    # ---- INIT HISTORY ----
    slot_history = [deque(maxlen=SMOOTH_WINDOW) for _ in polygons]

    last_refresh_ts = 0

    # ---- INFERENCE LOOP ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, TARGET_DIM)
        now = time.time()

        if now - last_refresh_ts >= OCCUPANCY_REFRESH_INTERVAL:
            last_refresh_ts = now

            results = model(frame, classes=[2, 7], conf=0.6, verbose=False)[0]
            slot_status = [False] * len(polygons)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                for i, poly in enumerate(polygons):
                    if point_in_poly((cx, cy), poly):
                        slot_status[i] = True
                        break

            smoothed = []
            for i, s in enumerate(slot_status):
                slot_history[i].append(s)
                smoothed.append(
                    sum(slot_history[i]) >= (len(slot_history[i]) // 2 + 1)
                )

            # 🔥 UPDATE SHARED STATE (FIXED)
            slot_state["timestamp"] = now
            slot_state["slots"] = [
                {"id": i + 1, "occupied": smoothed[i]}
                for i in range(len(smoothed))
            ]

        # ---- VISUALIZATION ----
        display = frame.copy()
        for i, poly in enumerate(polygons):
            occupied = sum(slot_history[i]) >= (len(slot_history[i]) // 2 + 1)
            color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(display, [poly], True, color, 3)

        cv2.imshow("Parking Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
