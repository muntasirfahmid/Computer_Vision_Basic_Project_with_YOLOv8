import cvzone
from ultralytics import YOLO
import cv2
import math

#for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# for recorded video footage
cap = cv2.VideoCapture("ppe3.mp4")

model = YOLO("ppe.pt")

cls_id = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
mycolor = (0,0,255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            ## FOR openCV ##
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            cls_id = int(box.cls[0])  # Class ID
            class_name = model.names[cls_id]
            label = f'{class_name} {conf:.2f}'
            if conf>0.5:
                if class_name == 'NO-Hardhat' or  class_name == 'NO-Safety Vest' or class_name == 'NO-Mask':
                    mycolor = (0,0,255)
                elif class_name == 'Hardhat' or class_name == 'Safety Vest' or class_name == 'Mask':
                        mycolor = (0,255,0)
                else:
                    mycolor = (255,0,0)


            cvzone.putTextRect(img, label, (max(0,x1),max(35,y1)),
                               scale=1,thickness=1,colorB= mycolor, colorT=(255,255,255),colorR=mycolor)
            cv2.rectangle(img, (x1,y1),(x2,y2), mycolor, 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)










