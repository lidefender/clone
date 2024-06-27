from ultralytics import YOLO
import cv2
import os
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0) # 0表示调用摄像头

# 逐一读取文件的图片并进行预测
# for i,img in enumerate(model.imgs):
#     # img = cv2.imread(img_path)
#     results = model(img)
#     annotated_frame = results[0].plot()
#     cv2.imshow('YOLOv8', annotated_frame)
#     if cv2.waitKey(1) == ord('q'):  # 按下q退出
#         break
# ressult = model()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8', annotated_frame)
    if cv2.waitKey(1) == ord('q'):  # 按下q退出
        break

cap.release()
cv2.destroyAllWindows()

