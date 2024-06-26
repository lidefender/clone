import jsomtest
import os
import numpy as np
import cv2
from labelme import utils
from PIL import Image

def json_to_mask(json_file, output_dir):
    """
    将单个JSON文件转换为掩码图像。

    参数：
    json_file (str): JSON文件的路径。
    output_dir (str): 输出掩码图像的目录。
    """
    # 读取JSON文件
    with open(json_file) as f:
        data = json.load(f)

    # 获取图像尺寸
    img_height = data['imageHeight']
    img_width = data['imageWidth']

    # 初始化空白掩码
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 标签名称和对应的索引
    label_to_index = {}
    index = 1

    # 遍历JSON文件中的每个标注形状
    for shape in data['shapes']:
        label = shape['label']
        if label not in label_to_index:
            label_to_index[label] = index
            index += 1

        # 将多边形转换为掩码
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], label_to_index[label])

    # 保存掩码图像
    mask_img = Image.fromarray(mask)
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    mask_img.save(os.path.join(output_dir, f'{base_name}_mask.png'))

    # 保存标签信息
    with open(os.path.join(output_dir, f'{base_name}_labels.txt'), 'w') as f:
        for label, index in label_to_index.items():
            f.write(f'{label}: {index}\n')

def process_folder(input_dir, output_dir):
    """
    处理整个文件夹中的所有JSON文件。

    参数：
    input_dir (str): 包含JSON文件的输入目录。
    output_dir (str): 输出掩码图像的目录。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_file = os.path.join(input_dir, filename)
            json_to_mask(json_file, output_dir)

if __name__ == "__main__":

    input_dir = r"F:\work\dataset\rebar2D\train2\label"
    output_dir = r"F:\work\dataset\rebar2D\train2\mask"

    process_folder(input_dir, output_dir)
