# OpenCV Object Detection and Tracking Projects

This repository contains **five different computer vision projects** built using **Python** and **OpenCV**.  
Each project demonstrates a different technique for **object detection** and **object tracking** using classical computer vision methods.

---

# 1) Body Detection from Image (Haar Cascade)

## Description
This project detects full human bodies in a static image using the Haar Cascade classifier provided by OpenCV.

## Features
- Loads an image from disk  
- Converts the image to grayscale  
- Detects human bodies using Haar Cascade  
- Draws bounding boxes around detected bodies  

## Technologies
- Python  
- OpenCV  
- Haar Cascade Classifier  

## How to Run
1. Place an image inside the project folder  
2. Update the image path in the code  
3. Run the script  

---

# 2) Real-Time Body Detection from Camera (Haar Cascade)

## Description
This project performs real-time human body detection using a webcam feed.

## Features
- Captures live video from webcam  
- Converts frames to grayscale  
- Detects human bodies in real time  
- Draws bounding boxes around detected bodies  
- Displays live FPS information  

## Technologies
- Python  
- OpenCV  
- Haar Cascade Classifier  

## How to Run
1. Connect a webcam  
2. Run the script  
3. Press **Q** to exit  

---

# 3) Car Detection from Image (Haar Cascade)

## Description
This project detects cars in a static image using a Haar Cascade car classifier.

## Features
- Loads an image from disk  
- Converts the image to grayscale  
- Detects cars using Haar Cascade  
- Draws bounding boxes around detected cars  

## Technologies
- Python  
- OpenCV  
- Haar Cascade Classifier  

## How to Run
1. Place a car image inside the project folder  
2. Update the image path in the code  
3. Run the script  

---

# 4) Car Detection from Video (Haar Cascade)

## Description
This project detects cars in a video file frame by frame using Haar Cascade.

## Features
- Loads a video file  
- Processes frames one by one  
- Converts frames to grayscale  
- Detects cars using Haar Cascade  
- Draws bounding boxes around detected cars  
- Displays detection results in real time  

## Technologies
- Python  
- OpenCV  
- Haar Cascade Classifier  

## How to Run
1. Place a video file inside the project folder  
2. Update the video path in the code  
3. Run the script  
4. Press **Q** to exit  

---

# 5) Object Tracking from Video (CSRT Tracker)

## Description
This project tracks a user-selected object in a video using the CSRT tracking algorithm.

## Features
- Loads a video file  
- Allows the user to select an object (ROI)  
- Tracks the selected object across frames  
- Draws a bounding box around the tracked object  
- Displays tracking status and FPS  

## Technologies
- Python  
- OpenCV  
- CSRT Tracker  
- NumPy  

## How to Run
1. Place a video file inside the project folder  
2. Run the script  
3. Select an object using the mouse  
4. Press **Q** to exit  

---

## Installation

```bash
pip install opencv-python numpy
