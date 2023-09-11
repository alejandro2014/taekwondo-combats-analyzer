from ultralytics import YOLO
import cv2

from body_drawer import BodyDrawer

model_path = './yolov8n-pose.pt'
image_path = './taekwondo.jpg'

img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]

body_drawer = BodyDrawer()

for i, result in enumerate(results):    
    points = result.keypoints.xy[0]

    img = body_drawer.draw(img, points)

cv2.imshow('result', img)
cv2.waitKey(0)