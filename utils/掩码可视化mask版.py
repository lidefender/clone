import os
import numpy as np
import cv2
from tqdm import tqdm

def apply_color_to_mask(mask, class_info):
    '''
    输入单通道灰度掩码图像，输出彩色掩码图像
    '''
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # 创建彩色空白图像

    for one_class in class_info:
        colored_mask[mask == one_class['label']] = one_class['color']

    return colored_mask

def overlay_masks_on_images(mask_path, images_path, masks_dir, class_info):
    '''
    输入原始图像路径和掩码图像路径，输出彩色掩码叠加图像
    '''
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for img_filename in tqdm(os.listdir(images_path)):
        try:
            img_path = os.path.join(images_path, img_filename)
            mask_filename = img_filename.split('.')[0] + '.png'
            mask_img_path = os.path.join(masks_dir, mask_filename)

            original_img = cv2.imread(img_path)
            mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"掩码文件 {mask_filename} 不存在或无法读取")
                continue

            colored_mask = apply_color_to_mask(mask, class_info)

            # 将彩色掩码与原图进行叠加
            overlay_img = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)

            overlay_output_path = os.path.join(mask_path, img_filename.split('.')[0] + '_overlay.png')
            cv2.imwrite(overlay_output_path, overlay_img)
        except Exception as e:
            print(img_filename, '转换失败', e)

if __name__ == '__main__':
    class_info = [
        {'label': 1, 'color': (0, 255, 0)},  # 绿色
        {'label': 2, 'color': (255, 0, 0)},  # 红色
        # 可以根据需要添加更多类别
    ]

    images_path = r"F:\work\dataset\rebar2D\train2\img"
    masks_dir = r"F:\work\dataset\rebar2D\train2\mask"  # 存放灰度掩码的文件夹
    overlay_output_dir = r"F:\work\dataset\rebar2D\train2\overlay"  # 存放叠加图像的文件夹

    overlay_masks_on_images(overlay_output_dir, images_path, masks_dir, class_info)
