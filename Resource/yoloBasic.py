from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model("Resource/img_2.png", show=True)
cv2.waitKey(0)