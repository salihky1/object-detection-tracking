import cv2
import numpy as np
import os
import time

video_path = "cars.mp4"
cascade_path = "haarcascade_car.xml"
window_name = "Car Detection Video"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video could not be opened")
    exit()

if not os.path.exists(cascade_path):
    print("Cascade file not found")
    exit()

car_cascade = cv2.CascadeClassifier(cascade_path)

if car_cascade.empty():
    print("Cascade could not be loaded")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

resize_scale = 100
new_width = int(frame_width * resize_scale / 100)
new_height = int(frame_height * resize_scale / 100)

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

    cars = car_cascade.detectMultiScale(
        gray_eroded,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in cars:
        center_x = x + w // 2
        center_y = y + h // 2
        radius = int((w + h) * 0.25)

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame_resized, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.putText(frame_resized, "CAR", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    current_time = time.time()
    fps_value = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time

    cv2.putText(frame_resized, f"FPS: {int(fps_value)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_resized, f"Resolution: {new_width}x{new_height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow(window_name, frame_resized)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        cv2.imwrite("saved_frame.jpg", frame_resized)

cap.release()
cv2.destroyAllWindows()
