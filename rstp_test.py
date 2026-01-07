import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import time
import cv2

url = "rtsp://127.0.0.1:8554/parking"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("FAILED to open RTSP stream")
    exit(1)

print("RTSP stream opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        time.sleep(1)
        continue

    cv2.imshow("RTSP Test", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
