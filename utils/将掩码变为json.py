import cv2  # 导入OpenCV库
import json  # 导入JSON库
import numpy as np  # 导入NumPy库
import os  # 导入os库，用于文件和目录操作
from 掩码变为白色 import *
def generate_json_from_mask(mask_path, json_path, label):
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码图像
    mask = np.array(intowrith(mask_path))
    print(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找图像中的轮廓

    if len(contours) == 0:  # 确保找到至少一个轮廓
        raise ValueError(f"No contours found in {mask_path}")

    contour = max(contours, key=cv2.contourArea)  # 只取最大的轮廓
    contour_list = contour.squeeze().tolist()  # 将轮廓坐标转换为列表

    if isinstance(contour_list[0], int):  # 检查并处理单点情况
        contour_list = [contour_list]

    # 创建标签数据结构
    data = {
        "shapes": [
            {
                "label": label,
                "points": contour_list,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": mask_path,
        "imageData": None,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }

    with open(json_path, 'w') as json_file:  # 将数据写入JSON文件
        json.dump(data, json_file, indent=4)

def process_masks_in_directory(directory, label):
    for filename in os.listdir(directory):  # 遍历目录中的所有文件
        if filename.endswith('.png'):  # 只处理PNG文件
            mask_path = os.path.join(directory, filename)  # 构建掩码图像路径
            json_path = os.path.splitext(mask_path)[0] + '.json'  # 构建JSON文件路径
            print(json_path)
            generate_json_from_mask(mask_path, json_path, label)  # 调用函数生成JSON文件

# 示例使用
directory = r"F:\work\dataset\rebar2D\train\mask1"   # 掩码图像文件夹路径
# mask_path = r"F:\work\dataset\rebar2D\train\TEMP\Image_20240622152022155.png"  # 掩码图像路径
# json_path = r"F:\work\dataset\rebar2D\train\jsontest\jsontest.json"  # 输出JSON文件路径
label = 'rebar'  # 目标标签

process_masks_in_directory(directory, label)  # 处理目录中的所有掩码文件
