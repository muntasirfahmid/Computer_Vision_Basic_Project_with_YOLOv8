import cvzone
import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *

# Load video footage
cap = cv2.VideoCapture("../Resource/people.mp4")

# Load YOLO model
model = YOLO('yolov8l.pt')

# Load mask for region of interest
mask = cv2.imread("pepMask.png")

# Initialize SORT tracker with defined parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define line limits for counting vehicles
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

while True:
    # Read frames from the video
    success, img = cap.read()

    # Apply mask to focus on the region of interest
    imgRegion = cv2.bitwise_and(img, mask)

    # Load and overlay graphical elements (if needed)
    imgGraphics = cv2.imread("upDown.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    # Get detection results from the YOLO model
    results = model(imgRegion, stream=True)

    # Initialize array for storing detection boxes
    detections = np.empty((0, 5))

    # Process each detection result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Get confidence score of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get the class ID and name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Filter classes and confidence threshold
            if class_name in "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with the detected objects
    resultsTracker = tracker.update(detections)

    # Draw a line for vehicle counting
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    # Process each tracked object
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box and ID for each tracked object
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Calculate the center of the object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the object crosses the defined line
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # # Display the total vehicle count on the frame
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    # Display the resulting image
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)

    # Wait for 1 ms before displaying the next frame
    cv2.waitKey(1)










