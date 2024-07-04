import cv2
import os
from pathlib import Path
from ultralytics.data.annotator import auto_annotate


def process_video(video_path, output_dir, det_model, sam_model, frame_interval=1, manual_roi=False):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 仅处理每隔 frame_interval 的帧
        if frame_count % frame_interval == 0:
            if manual_roi:
                # 显示帧以手动选择ROI
                r = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
                roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                cv2.destroyWindow("Select ROI")
            else:
                roi = frame

            # 保存当前帧
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, roi)

            # 自动标注
            auto_annotate(data=frame_filename, det_model=det_model, sam_model=sam_model, output_dir=output_dir)

        frame_count += 1

    cap.release()


# 使用示例

video_path = r"F:\work\python\clone\2d\ultralnew\ultralytics\dataset\v1.mp4"
output_dir = r"F:\work\dataset\rebar2D\train\video\annotated_frames"
det_model = r"F:\warehouse\download\epoch160.pt"
sam_model = r"F:\warehouse\download\sam_b.pt"

# 处理视频，每隔10帧标注一帧，并且不使用手动ROI选择
process_video(video_path, output_dir, det_model, sam_model, frame_interval=60, manual_roi=True)

