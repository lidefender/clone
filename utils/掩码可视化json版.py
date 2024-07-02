import os
import json
import numpy as np
import cv2
from tqdm import tqdm

def labelme2colored_mask_single_img(img_path, labelme_json_path, class_info):
    '''
    输入原始图像路径和labelme标注路径，输出彩色掩码图像
    '''
    img_bgr = cv2.imread(img_path)
    img_mask = np.zeros_like(img_bgr)  # 创建彩色空白图像

    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    for one_class in class_info:  # 按顺序遍历每一个类别
        for each in labelme['shapes']:  # 遍历所有标注，找到属于当前类别的标注
            if each['label'] == one_class['label']:
                if one_class['type'] == 'polygon':  # polygon 多段线标注
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]
                    img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])

                elif one_class['type'] == 'line' or one_class['type'] == 'linestrip':  # line 或者 linestrip 线段标注
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]
                    img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'],
                                             thickness=one_class.get('thickness', 1))

                elif one_class['type'] == 'circle':  # circle 圆形标注
                    points = np.array(each['points'], dtype=np.int32)
                    center_x, center_y = points[0]
                    edge_x, edge_y = points[1]
                    radius = np.linalg.norm(np.array([center_x, center_y]) - np.array([edge_x, edge_y])).astype('uint32')
                    img_mask = cv2.circle(img_mask, (center_x, center_y), radius, one_class['color'], one_class.get('thickness', -1))

                else:
                    print('未知标注类型', one_class['type'])

    return img_mask

def overlay_masks_on_images(mask_path, images_path, labelme_json_dir, class_info):
    '''
    输入原始图像路径和labelme标注路径，输出彩色掩码叠加图像
    '''
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for img_filename in tqdm(os.listdir(images_path)):
        try:
            img_path = os.path.join(images_path, img_filename)
            json_filename = '.'.join(img_filename.split('.')[:-1]) + '.json'
            labelme_json_path = os.path.join(labelme_json_dir, json_filename)

            colored_mask = labelme2colored_mask_single_img(img_path, labelme_json_path, class_info)
            original_img = cv2.imread(img_path)

            # 将彩色掩码与原图进行叠加
            overlay_img = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)

            mask_output_path = os.path.join(mask_path, img_filename.split('.')[0] + '_overlay.png')
            cv2.imwrite(mask_output_path, overlay_img)
        except Exception as e:
            print(img_filename, '转换失败', e)

if __name__ == '__main__':
    class_info = [
        {'label': 'rebar', 'type': 'polygon', 'color': (0, 255, 0)},  # 绿色
        {'label': 'socket', 'type': 'polygon', 'color': (255, 0, 0)},  # 红色
        # 可以根据需要添加更多类别
    ]

    images_path = r"F:\work\dataset\rebar2D\train2\img"
    labelme_json_dir = r"F:\work\dataset\rebar2D\train2\label2"
    mask_path = r"F:\work\dataset\rebar2D\train2\TEMP"

    overlay_masks_on_images(mask_path, images_path, labelme_json_dir, class_info)
