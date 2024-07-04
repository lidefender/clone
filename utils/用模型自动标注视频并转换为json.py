import cv2
import os
import json
from pathlib import Path
from ultralytics.data.annotator import auto_annotate
from PIL import Image


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
    video_name = os.path.splitext(os.path.basename(video_path))[0]

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
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, roi_frame)

            # 自动标注
            auto_annotate(data=frame_filename, det_model=det_model, sam_model=sam_model, output_dir=output_dir)

            # 检查是否生成了YOLO标签文件
            yolo_label_path = os.path.splitext(frame_filename)[0] + '.txt'
            if os.path.exists(yolo_label_path):
                json_label_path = os.path.splitext(frame_filename)[0] + '.json'
                # 定义标签映射
                label_map = {
                    0: 'rebar',
                    1: 'socket'
                    # 添加更多标签映射
                }
                # 转换为JSON格式
                yolo_to_json(yolo_label_path, json_label_path, frame_filename, label_map)

        frame_count += 1

    cap.release()
    if show_video:
        cv2.destroyAllWindows()


def yolo_to_json(yolo_file_path, json_file_path, image_path, label_map):
    # 读取图片以获取宽度和高度
    image = Image.open(image_path)
    image_width, image_height = image.size

    shapes = []
    with open(yolo_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_index = int(parts[0])
            label = label_map.get(class_index, str(class_index))  # 获取自定义标签名称
            points = []
            for i in range(1, len(parts), 2):
                x = float(parts[i]) * image_width
                y = float(parts[i + 1]) * image_height
                points.append([x, y ])
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)

    json_data = {
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


# 主程序入口
if __name__ == '__main__':
    video_path = r"F:\work\python\clone\2d\ultralnew\ultralytics\dataset\tensilestrengthtesting.mp4"
    output_dir = r"F:\work\dataset\rebar2D\train\video\annotated_frames"
    det_model = r"F:\warehouse\download\epoch160.pt"
    sam_model = r"F:\warehouse\download\sam_b.pt"
    frame_interval = 20
    manual_roi = True
    show_video = False

    # 处理视频，每隔10帧标注一帧，并且使用手动ROI选择和实时显示视频
    process_video(video_path, output_dir, det_model, sam_model, frame_interval, manual_roi, show_video)
