import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import deque

TARGET_DIM = (1024, 768)
current_polygon = []
polygons = []

slot_state = {
    "timestamp": None,
    "slots": []  
}

SMOOTH_WINDOW = 3 

slot_history = []  

def mouse_callback(event, x, y, flags, param):
    global current_polygon, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            polygons.append(np.array(current_polygon, dtype=np.int32))
        current_polygon.clear()

def draw_polygons(frame):
    for idx, poly in enumerate(polygons):
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        M = cv2.moments(poly)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(frame, f"Slot {idx + 1}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if len(current_polygon) > 1:
        pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

def notify_admin(alert):
    print("ADMIN ALERT:", alert)

def main(video_path="resources/AED Parking Video.mp4"):
    global polygons, current_polygon, slot_history

    model = YOLO('yolov8s.pt')  

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return

    first_frame_resized = cv2.resize(first_frame, TARGET_DIM)
    cv2.namedWindow('Define Parking Slots', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Define Parking Slots', mouse_callback)

    while True:
        display_frame = first_frame_resized.copy()
        draw_polygons(display_frame)

        cv2.putText(display_frame, "Left click: add point | Right click: close slot | 's': start | 'q': quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Define Parking Slots', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(current_polygon) >= 3:
                polygons.append(np.array(current_polygon, dtype=np.int32))
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow('Define Parking Slots')

    slot_history = [deque(maxlen=SMOOTH_WINDOW) for _ in polygons]

    last_refresh_ts = 0
    OCCUPANCY_REFRESH_INTERVAL = 5.0  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_resized = cv2.resize(frame, TARGET_DIM)

        now = time.time()
        if now - last_refresh_ts >= OCCUPANCY_REFRESH_INTERVAL:
            last_refresh_ts = now

            results = model(frame_resized, classes=[2,7], conf=0.6, verbose=False)[0]

            slot_status = [False] * len(polygons)
            detected_centers = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                detected_centers.append((center_x, center_y))

                for i, poly in enumerate(polygons):
                    if point_in_poly((center_x, center_y), poly):
                        slot_status[i] = True
                        break

            smoothed_status = []
            for i, s in enumerate(slot_status):
                slot_history[i].append(s)
                # majority vote
                true_count = sum(slot_history[i])
                smoothed = true_count >= (len(slot_history[i]) // 2 + 1) if len(slot_history[i]) > 0 else s
                smoothed_status.append(smoothed)

            
            threading.Thread(target=notify_admin, args=({'slots': smoothed_status, 'ts': now},), daemon=True).start()

            last_detected_centers = detected_centers

        display = frame_resized.copy()
        try:
            for i, poly in enumerate(polygons):
                occupied = slot_history[i] and sum(slot_history[i]) >= (len(slot_history[i]) // 2 + 1)
                color = (0, 0, 255) if occupied else (0, 255, 0)
                status_text = 'Occupied' if occupied else 'Free'
                cv2.polylines(display, [poly], isClosed=True, color=color, thickness=3)
                M = cv2.moments(poly)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(display, f"Slot {i + 1}: {status_text}", (cx - 50, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            pass

        if 'last_detected_centers' in locals():
            for c in last_detected_centers:
                cv2.circle(display, c, 5, (255, 0, 0), -1)

        cv2.imshow('Parking Detection', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
