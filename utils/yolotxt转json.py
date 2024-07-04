import json
import os
from PIL import Image


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
                points.append([x, y-3])
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


def batch_process(label_folder, image_folder, output_folder, label_map):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            yolo_file_path = os.path.join(label_folder, label_file)
            image_file_path = os.path.join(image_folder, base_name + '.jpg')
            json_file_path = os.path.join(output_folder, base_name + '.json')
            yolo_to_json(yolo_file_path, json_file_path, image_file_path, label_map)
            # if os.path.exists(image_file_path):
            #     yolo_to_json(yolo_file_path, json_file_path, image_file_path, label_map)
            # else:
            #     print(f"Image file {image_file_path} not found for label {label_file}")


# 示例调用
if __name__ == '__main__':
    label_folder = r"F:\work\dataset\rebar2D\train\video\annotated_frames"
    image_folder = r"F:\work\dataset\rebar2D\train\video\annotated_frames"
    output_folder = r"F:\work\dataset\rebar2D\train\video\annotated_frames"

    # 定义标签映射
    label_map = {
        0: 'rebar',
        1: 'socket'

        # 添加更多标签映射
    }

    batch_process(label_folder, image_folder, output_folder, label_map)