import os
import json
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from PIL import Image



def labelme2mask_single_img(img_path, labelme_json_path):
    '''
    输入原始图像路径和labelme标注路径，输出单通道灰度 mask
    '''

    img_bgr = cv2.imread(img_path)
    img_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)  # 创建单通道空白图像

    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    for one_class in class_info:  # 按顺序遍历每一个类别
        for each in labelme['shapes']:  # 遍历所有标注，找到属于当前类别的标注
            if each['label'] == one_class['label']:
                if one_class['type'] == 'polygon':  # polygon 多段线标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（闭合区域）
                    img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])

                elif one_class['type'] == 'line' or one_class['type'] == 'linestrip':  # line 或者 linestrip 线段标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（非闭合区域）
                    img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'],
                                             thickness=one_class['thickness'])

                elif one_class['type'] == 'circle':  # circle 圆形标注

                    points = np.array(each['points'], dtype=np.int32)

                    center_x, center_y = points[0][0], points[0][1]  # 圆心点坐标

                    edge_x, edge_y = points[1][0], points[1][1]  # 圆周点坐标

                    radius = np.linalg.norm(np.array([center_x, center_y] - np.array([edge_x, edge_y]))).astype(
                        'uint32')  # 半径

                    img_mask = cv2.circle(img_mask, (center_x, center_y), radius, one_class['color'],
                                          one_class['thickness'])

                else:
                    print('未知标注类型', one_class['type'])

    return img_mask

# class_info = [
#     {'label':'rebar', 'type':'polygon', 'color':1},
#                                                     # polygon 多段线
#
# ]

# Dataset_Path = r'F:\work\python\clone\dataset\rebar2d' # 数据集路径
# os.chdir(r"F:\work\dataset\rebar2D\train\TEMP")   # 掩码图像文件夹路径

def labelme2mask(mask_path, images_path, labelme_json_path):
    '''
    输入原始图像路径和labelme标注路径，输出单通道灰度 mask
    '''

    for img_path in tqdm(os.listdir(images_path)):

        try:
            labelme_json_path = os.path.join(labelme_json_path,'.'.join(img_path.split('.')[:-1]) + '.json')
            img_mask = labelme2mask_single_img(os.path.join(images_path,img_path), labelme_json_path)

            mask_path1 = os.path.join(mask_path,img_path.split('.')[0] + '.png')
            print(mask_path)

            # 将单通道灰度图像保存
            cv2.imwrite(mask_path1, img_mask)

        except Exception as E:
            print(img_path, '转换失败', E)


if __name__ == '__main__':
    class_info = [
        {'label': 'rebar', 'type': 'polygon', 'color': 1}
        # polygon 多段线

    ]

    Dataset_Path = r"F:\work\dataset\rebar2D\train"  # 数据集路径
    img_path = r"D:\work\dataset\rebar2D\train\img2_auto_annotate_labels"
    labelme_json_path = r"D:\work\dataset\rebar2D\train\jsontest"
    mask_path=r"F:\work\dataset\rebar2D\train\TEMP" # 掩码图像文件夹路径
    labelme2mask(mask_path, img_path, labelme_json_path)