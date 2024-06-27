from ultralytics import YOLO
import cv2
import os
model = YOLO(r"F:\work\python\yolo\code\ultralytics\runs\detect\train19\weights\best.pt")

for i in range(len(os.listdir(os.getcwd()))):
    if os.listdir(os.getcwd())[i].split('.')[-1] == 'jpg' or os.listdir(os.getcwd())[i].split('.')[-1] == 'png':
        img_
path=os.path.join(os.getcwd(),os.listdir(os.getcwd())[i])
# cap = cv2.VideoCapture(0) # 0表示调用摄像头

# 逐一读取文件的图片并进行预测
for i,img in enumerate(model.imgs):
    # img = cv2.imread(img_path)
    results = model(img)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8', annotated_frame)
    if cv2.waitKey(1) == ord('q'):  # 按下q退出
        break
ressult = model()
