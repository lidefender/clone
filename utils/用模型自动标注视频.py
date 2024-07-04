import cv2
import os
from pathlib import Path
from ultralytics.data.annotator import auto_annotate


def process_video(video_path, output_dir, det_model, sam_model, frame_interval=1, manual_roi=False, show_video=False):
    """
    处理视频文件，自动标注每隔 frame_interval 帧提取的帧图像，并根据需要进行手动选择ROI。

    参数:
        video_path (str): 视频文件路径。
        output_dir (str): 输出目录，用于保存标注后的图像和标签数据。
        det_model (str): 预训练的YOLO检测模型路径。
        sam_model (str): 预训练的SAM分割模型路径。
        frame_interval (int, optional): 每隔多少帧提取一帧进行标注。默认值为1。
        manual_roi (bool, optional): 是否进行手动选择ROI。默认值为False。
        show_video (bool, optional): 是否实时显示视频并允许按R键重新选择ROI。默认值为False。

    示例:
        video_path = r"F:/work/dataset/rebar2D/train/video/input_video.mp4"
        output_dir = r"F:/work/dataset/rebar2D/train/video/annotated_frames"
        det_model = r"F:/warehouse/download/best.pt"
        sam_model = r"F:/warehouse/download/sam_b.pt"
        process_video(video_path, output_dir, det_model, sam_model, frame_interval=10, manual_roi=True, show_video=True)
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    roi = None

    if manual_roi:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频帧以选择ROI")
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开头

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if show_video:
            cv2.imshow("Video", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('r') and manual_roi:  # 按 'r' 键重新选择ROI
                roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select ROI")
            elif k == ord('q'):  # 按 'q' 键退出
                break

        # 仅处理每隔 frame_interval 的帧
        if frame_count % frame_interval == 0:
            if manual_roi and roi is not None and all(roi):
                roi_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            else:
                roi_frame = frame

            # 保存当前帧
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, roi_frame)

            # 自动标注
            auto_annotate(data=frame_filename, det_model=det_model, sam_model=sam_model, output_dir=output_dir)

        frame_count += 1

    cap.release()
    if show_video:
        cv2.destroyAllWindows()


# 使用示例
video_path = r"F:\work\python\clone\2d\ultralnew\ultralytics\dataset\32mm coupler steel bar tensile strength testing.mp4"
output_dir = r"F:\work\dataset\rebar2D\train\video\annotated_frames"
det_model = r"F:\warehouse\download\best (1).pt"
sam_model = r"F:\warehouse\download\sam_b.pt"

# 处理视频，每隔10帧标注一帧，并且使用手动ROI选择和实时显示视频
process_video(video_path, output_dir, det_model, sam_model, frame_interval=10, manual_roi=True, show_video=True)
