import cv2
import numpy as np

# 设置初始标记点（根据具体视频调整坐标）
initial_point1 = (100, 200)  # 示例坐标1
initial_point2 = (400, 200)  # 示例坐标2

# 计算初始标距
initial_gauge_length = np.linalg.norm(np.array(initial_point2) - np.array(initial_point1))

# 打开视频捕获对象（0 表示摄像头，也可以替换为视频文件路径）
cap = cv2.VideoCapture('tensile_test_video.mp4')  # 替换为实际视频文件路径

# 检查是否成功打开视频流
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 在当前帧中绘制初始标记点
    cv2.circle(frame, initial_point1, 5, (0, 255, 0), -1)
    cv2.circle(frame, initial_point2, 5, (0, 255, 0), -1)

    # 计算当前标记点的位置（这里假设标记点在图像处理后可以检测到，这里用初始点模拟）
    # 在实际应用中，可以使用图像处理或机器学习方法检测标记点
    current_point1 = initial_point1  # 模拟当前点1
    current_point2 = (initial_point2[0] + 10, initial_point2[1])  # 模拟当前点2，假设有伸长

    # 计算当前标距
    current_gauge_length = np.linalg.norm(np.array(current_point2) - np.array(current_point1))

    # 计算伸长量
    elongation = current_gauge_length - initial_gauge_length

    # 在视频帧上显示伸长量
    cv2.putText(frame, f'Elongation: {elongation:.2f} pixels', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示处理后的帧
    cv2.imshow('Tensile Test Visualization', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象
cap.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
