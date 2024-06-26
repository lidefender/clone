import cv2
import os
import numpy as np

input_folder = r'F:\work\dataset\rebar2D\train\img'
output_folder = r'F:\work\dataset\rebar2D\train\mask'
def process_images(input_folder, output_folder, object_label=1, background_label=0):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # 转换为灰度图像
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 应用Otsu的自动阈值二值化
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow("g", gray_image)
            # cv2.waitKey(0)
            # 将二值图像转换为掩码
            mask = np.where(binary_image == 255, object_label, background_label).astype(np.uint8)

            # 保存掩码图像
            mask_filename = os.path.splitext(filename)[0] + '_mask.png'
            mask_path = os.path.join(output_folder, mask_filename)
            cv2.imwrite(mask_path, mask)

    print("处理完成！")


# 设置输入文件夹和输出文件夹路径
# input_folder = 'path_to_input_folder'
# output_folder = 'path_to_output_folder'
input_folder = r'H:\data\rebar2D\train2\img'
output_folder = r'F:\work\dataset\rebar2D\train\mask'
# 调用函数处理图像
process_images(input_folder, output_folder, object_label=255, background_label=0)
