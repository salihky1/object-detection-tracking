import cv2
import numpy as np
import os
import time

video_path = "kosucular.mp4"
window_name = "Object Tracking Screen"

if not os.path.exists(video_path):
    print("Video file not found")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video could not be opened")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

resize_scale = 100
new_width = int(frame_width * resize_scale / 100)
new_height = int(frame_height * resize_scale / 100)

ret, frame = cap.read()
if not ret or frame is None:
    print("First frame could not be read")
    cap.release()
    exit()

frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

gray_init = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
gray_init = cv2.GaussianBlur(gray_init, (5, 5), 0)

bbox = cv2.selectROI(window_name, frame_resized, False)
if bbox is None or len(bbox) != 4:
    print("ROI selection failed")
    cap.release()
    exit()

tracker = cv2.TrackerCSRT_create()

tracker_initialized = tracker.init(frame_resized, bbox)
if not tracker_initialized:
    print("Tracker could not be initialized")
    cap.release()
    exit()

kernel = np.ones((3, 3), np.uint8)

previous_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_equalized = cv2.equalizeHist(gray_blur)

    gray_dilated = cv2.dilate(gray_equalized, kernel, iterations=1)
    gray_eroded = cv2.erode(gray_dilated, kernel, iterations=1)

    success, bbox = tracker.update(frame_resized)

    if success:
        x, y, w, h = [int(i) for i in bbox]
        center_x = x + w // 2
        center_y = y + h // 2

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame_resized, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(frame_resized, "TRACKING", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame_resized, "TRACK LOST", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    current_time = time.time()
    fps_value = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time

    cv2.putText(frame_resized, f"FPS: {int(fps_value)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame_resized, f"Resolution: {new_width}x{new_height}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(window_name, frame_resized)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        bbox = cv2.selectROI(window_name, frame_resized, False)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame_resized, bbox)

cap.release()
cv2.destroyAllWindows()
