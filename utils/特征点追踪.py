import cv2
import numpy as np

# 读取视频
video_path = 'rebar_video.mp4'
cap = cv2.VideoCapture(video_path)

# 视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = total_frames / fps

# 截取时间段（单位：秒）
start_time = 5  # 开始时间
end_time = 10  # 结束时间

# 计算开始帧和结束帧
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# 设置视频开始帧
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# ShiTomasi角点检测参数
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade光流法参数
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# 随机颜色
color = np.random.randint(0, 255, (200, 3))

# 获取第一帧
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 在第一帧中找到特征点
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个掩码图像用于绘制轨迹
mask = np.zeros_like(old_frame)

# 处理指定时间段内的每一帧
while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择好的特征点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 更新旧帧和旧特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()