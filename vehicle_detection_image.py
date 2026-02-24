import cv2
import numpy as np
import os

image_path = "car.jpg"
cascade_path = "haarcascade_car.xml"

image = cv2.imread(image_path)
if image is None:
    print("Image could not be loaded")
    exit()

if not os.path.exists(cascade_path):
    print("Cascade file not found")
    exit()

car_cascade = cv2.CascadeClassifier(cascade_path)
if car_cascade.empty():
    print("Cascade could not be loaded")
    exit()

original_height, original_width = image.shape[:2]

scale_percent = 100
new_width = int(original_width * scale_percent / 100)
new_height = int(original_height * scale_percent / 100)

image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

gray_equalized = cv2.equalizeHist(gray_blur)

kernel = np.ones((3, 3), np.uint8)

gray_dilated = cv2.dilate(gray_equalized, kernel, iterations=1)

gray_eroded = cv2.erode(gray_dilated, kernel, iterations=1)

cars = car_cascade.detectMultiScale(
    gray_eroded,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in cars:
    center_x = x + w // 2
    center_y = y + h // 2
    radius = int((w + h) * 0.25)

    cv2.rectangle(image_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(image_resized, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.putText(image_resized, "CAR", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

output_image_path = "car_detection_result.jpg"
cv2.imwrite(output_image_path, image_resized)

window_original = "Original Image"
window_gray = "Gray Image"
window_processed = "Processed Image"
window_result = "Car Detection Result"

cv2.imshow(window_original, image)
cv2.imshow(window_gray, gray)
cv2.imshow(window_processed, gray_eroded)
cv2.imshow(window_result, image_resized)

key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
