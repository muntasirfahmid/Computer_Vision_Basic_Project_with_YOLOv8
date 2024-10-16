import cvzone
from ultralytics import YOLO
import cv2
import math

#for webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# for recorded video footage
# cap = cv2.VideoCapture("../Resource/bikes.mp4")

model = YOLO('yolov8n.pt')


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            ## FOR openCV ##
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            #print(x1, y1, x2, y2)

            ## FOR CVzone ##
            #x1, y1, w, h = box.xywh[0]
            w,h = x2-x1, y2-y1
            ##bbox= int(x1), int(y1), int(w),int(h)
            cvzone.cornerRect(img, (x1,y1,w,h))

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            cls_id = int(box.cls[0])  # Class ID
            class_name = model.names[cls_id]
            label = f'{class_name} {conf:.2f}'


            cvzone.putTextRect(img, label, (max(0,x1),max(35,y1)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)










