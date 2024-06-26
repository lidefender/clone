import jsomtest
import os
import numpy as np
import cv2
from labelme import utils
from PIL import Image


def json_to_mask(json_file, output_dir):
    # 读取JSON文件
    with open(json_file) as f:
        data = json.load(f)
    print(data)
    # 获取图像尺寸
    img_height = data['imageHeight']
    img_width = data['imageWidth']

    # 初始化空白掩码
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 标签名称和对应的索引
    label_to_index = {}
    index = 1

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


if __name__ == "__main__":
    # import sys
    #
    # if len(sys.argv) != 3:
    #     print("Usage: python json_to_mask.py <json_file> <output_dir>")
    #     sys.exit(1)
    #
    # json_file = sys.argv[1]
    # output_dir = sys.argv[2]
    #
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    json_file = r"F:\work\dataset\rebar2D\train2\label\Image_20240622152043066.json"
    output_dir = r"F:\work\dataset\rebar2D\train2\mask"
    json_to_mask(json_file, output_dir)
