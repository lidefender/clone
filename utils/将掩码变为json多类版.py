import os
import json
import numpy as np
import cv2
from PIL import Image

def get_polygons(mask, gray_value):
    # 找到特定灰度值的所有轮廓
    contours, _ = cv2.findContours((mask == gray_value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.reshape(-1, 2).tolist() for contour in contours if contour.shape[0] >= 3]
    return polygons

def process_image(mask_path):
    # 读取掩码图像
    mask = np.array(Image.open(mask_path))
    height, width = mask.shape[:2]

    # 灰度值到标签的映射
    gray_value_to_label = {
        1: "rebar",  # 示例映射，可以根据需要修改
        2: "socket",
    }

    shapes = []
    for gray_value, label in gray_value_to_label.items():
        polygons = get_polygons(mask, gray_value)
        for polygon in polygons:
            shape = {
                "label": label,
                "points": polygon,
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
        "imagePath": os.path.basename(mask_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    return json_data

def process_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            mask_path = os.path.join(folder_path, filename)
            json_data = process_image(mask_path)

            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder, json_filename)

            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_folder = r"F:\work\dataset\rebar2D\train2\mask2"
    output_folder = r"F:\work\dataset\rebar2D\train2\label2"
    process_folder(input_folder, output_folder)
