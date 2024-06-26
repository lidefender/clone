import os
import jsomtest
import numpy as np
from PIL import Image, ImageDraw


def create_mask_from_json(json_path, output_folder, image_size, object_label=1, background_label=0):
    # 读取JSON文件
    with open(json_path) as f:
        data = json.load(f)

    # 创建一个空白的掩码图像
    mask = Image.new('L', image_size, background_label)
    draw = ImageDraw.Draw(mask)

    # 遍历每个标注对象
    for annotation in data['annotations']:
        # 假设标注对象的多边形坐标存储在'polygon'键中
        polygon = annotation['polygon']

        # 画多边形
        draw.polygon(polygon, outline=object_label, fill=object_label)

    return mask


def process_folder(input_folder, output_folder, image_folder, object_label=1, background_label=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            image_filename = os.path.splitext(filename)[0] + '.jpg'  # 假设图像是jpg格式
            image_path = os.path.join(image_folder, image_filename)

            # 读取图像以获取图像大小
            image = Image.open(image_path)
            image_size = image.size  # (width, height)

            # 创建掩码
            mask = create_mask_from_json(json_path, output_folder, image_size, object_label, background_label)

            # 保存掩码图像
            mask_filename = os.path.splitext(filename)[0] + '_mask.png'
            mask_path = os.path.join(output_folder, mask_filename)
            mask.save(mask_path)


# 设置输入文件夹、图像文件夹和输出文件夹路径
input_folder = r'H:\data\rebar2D\train2\label'
image_folder = r'H:\data\rebar2D\train2\img'
output_folder = r'H:\data\rebar2D\train2\mask'

# 调用函数处理图像
process_folder(input_folder, output_folder, image_folder, object_label=1, background_label=0)
