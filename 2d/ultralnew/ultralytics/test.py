from ultralytics import YOLO
import cv2

# 加载预训练模型
model = YOLO(r"F:\warehouse\download\best (1).pt")

# 打开视频捕获对象（0 表示摄像头，也可以替换为视频文件路径）
cap = cv2.VideoCapture(r"D:\work\python\clone\2d\ultralnew\ultralytics\dataset\v1.mp4")

# 检查是否成功打开视频流
if not cap.isOpened():
    print("Error: Could not open video stream or file")

    exit()

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频编解码器并创建VideoWriter对象，用于保存视频
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 使用模型进行预测
    results = model(frame)
    for result in results:
        masks = result.masks  # 获取分割掩码
        frame = result.plot()  # 在帧上绘制分割结果

    # 显示处理后的帧
    cv2.imshow('YOLOv8 Segmentation', frame)

    # 将帧写入输出视频文件
    out.write(frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获和视频写入对象
cap.release()
out.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
