import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import albumentations as A  # 导入Albumentations库，用于图像增强
import os  # 导入OS库，用于文件和目录操作
import json  # 导入JSON库，用于处理JSON格式的数据


def pca_color_augmentation(image, alpha_std=0.1):
    """ 使用PCA对图像进行颜色增强 """
    orig_img = image.astype(float)  # 将图像转换为浮点数类型
    img_rs = orig_img.reshape(-1, 3)  # 将图像重塑为二维数组
    img_mean = np.mean(img_rs, axis=0)  # 计算每个通道的均值
    img_std = np.std(img_rs, axis=0)  # 计算每个通道的标准差
    img_rs = (img_rs - img_mean) / img_std  # 对图像进行标准化处理

    cov = np.cov(img_rs, rowvar=False)  # 计算协方差矩阵
    eigvals, eigvecs = np.linalg.eigh(cov)  # 计算特征值和特征向量

    noise = np.random.normal(0, alpha_std, 3)  # 生成正态分布的噪声
    noise = eigvecs @ (eigvals ** 0.5) * noise  # 应用PCA变换
    noise = (noise * img_std) + img_mean  # 反标准化处理

    aug_img = orig_img + noise  # 将噪声加到原图像上
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)  # 裁剪值并转换为8位无符号整数
    return aug_img  # 返回增强后的图像


import cv2
import numpy as np


def color_augment(image, alpha=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间

    # # 棕褐色的范围
    # lower_brown = np.array([10, 100, 100])  # 棕色的下限
    # upper_brown = np.array([20, 255, 200])  # 棕色的上限
    #
    # # 深棕色或黑色的范围
    # lower_dark_brown = np.array([0, 50, 50])  # 深棕色的下限
    # upper_dark_brown = np.array([20, 150, 150])  # 深棕色的上限
    #
    # mask = cv2.inRange(hsv, lower_brown, upper_brown)  # 创建棕色区域的掩码
    # dark_mask = cv2.inRange(hsv, lower_dark_brown, upper_dark_brown)  # 创建深棕色区域的掩码
    #
    # combined_mask = cv2.bitwise_or(mask, dark_mask)  # 合并两个掩码
    #
    # brown_tint = np.array([19, 69, 139], dtype=np.float32)  # 定义棕褐色的色调
    # image = image.astype(np.float32)
    # image[combined_mask > 0] = (1 - alpha) * image[combined_mask > 0] + alpha * brown_tint  # 仅对掩码区域进行颜色增强
    # image = np.clip(image, 0, 255).astype(np.uint8) ###
    #
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间
    # # 黑铁色的范围（深灰色到黑色）
    # lower_black_iron = np.array([0, 0, 0])  # 黑铁色的下限
    # upper_black_iron = np.array([180, 255, 70])  # 黑铁色的上限
    #
    # mask = cv2.inRange(hsv, lower_black_iron, upper_black_iron)  # 创建黑铁色区域的掩码
    #
    # silver_tint = silver_color()  # 获取亮银色
    # image = image.astype(np.float32)
    # image[mask > 0] = (1 - alpha) * image[mask > 0] + alpha * silver_tint  # 仅对掩码区域进行颜色增强
    # image = np.clip(image, 0, 255).astype(np.uint8)

    # 黑铁色的范围（深灰色到黑色）
    lower_black_iron = np.array([0, 0, 0])  # 黑铁色的下限
    upper_black_iron = np.array([180, 255, 70])  # 黑铁色的上限

    mask = cv2.inRange(hsv, lower_black_iron, upper_black_iron)  # 创建黑铁色区域的掩码

    # 创建亮银色的色调
    silver_tint = np.array([192, 192, 192], dtype=np.float32)

    # 将图像转换为浮点数格式，以便于计算
    image = image.astype(np.float32)

    # 仅对掩码区域进行颜色变换
    image[mask > 0] = (1 - alpha) * image[mask > 0] + alpha * silver_tint

    # 增加亮度和对比度以模拟金属光泽
    image[mask > 0] = cv2.addWeighted(image[mask > 0], 1.5, np.zeros(image[mask > 0].shape, image.dtype), 0, 0)

    # 限制像素值在0到255之间
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def silver_color():
    r=np.random.randint(0, 255)
    if r<124:
        color= np.array([192, 192, 192], dtype=np.float32)
    else:
        random_tint = np.random.randint(0, 256, size=3)  # 获取一个随机颜色
    # 定义亮银色
    color= np.array([192, 192, 192], dtype=np.float32)
    return color


def augment_image(image, mask, n_augmentations=1, color_augmentation='pca'):
    # 定义数据增强的变换序列
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转，概率为50%
        A.VerticalFlip(p=0.5),  # 垂直翻转，概率为50%
        A.RandomRotate90(p=0.5),  # 随机旋转90度，概率为50%
        A.Transpose(p=0.5),  # 转置操作（交换x和y轴），概率为50%
        A.RandomBrightnessContrast(p=0.2),  # 随机调整亮度和对比度，概率为20%
        A.ElasticTransform(p=0.2),  # 弹性变换，概率为20%
        A.GridDistortion(p=0.2),  # 网格扭曲，概率为20%
        A.ShiftScaleRotate(shift_limit=0.0825, scale_limit=0.1, rotate_limit=45, p=0.3),  # 平移、缩放和旋转，概率为20%
        A.RandomCrop(height=640, width=640, p=0.4),  # 随机裁剪到256x256大小，概率为100%
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),  # 颜色抖动，概率为30%
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # 高斯噪声，概率为20%
        A.MotionBlur(blur_limit=3, p=0.2),  # 模糊，概率为20%
        # A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.2)  # 随机遮挡，概率为20%
    ])

    augmented_images = []  # 存储增强后的图像
    augmented_masks = []  # 存储增强后的掩码

    for _ in range(n_augmentations):
        augmented = transform(image=image, mask=mask)  # 对图像和掩码进行增强
        image_aug = augmented['image']  # 获取增强后的图像
        mask_aug = augmented['mask']  # 获取增强后的掩码

        # 应用选择的颜色增强方法
        if color_augmentation == 'pca':
            image_aug = pca_color_augmentation(image_aug)
        elif color_augmentation == 'color':
            image_aug = color_augment(image_aug)

        augmented_images.append(image_aug)  # 添加到增强图像列表
        augmented_masks.append(mask_aug)  # 添加到增强掩码列表

    return augmented_images, augmented_masks  # 返回增强后的图像和掩码


def read_mask(mask_path, label_format):
    # 读取掩码文件，根据标签格式选择不同的方法
    if label_format == "mask":
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # 读取PNG格式掩码
    elif label_format == "json":
        with open(mask_path, 'r') as f:
            annotations = json.load(f)  # 读取JSON文件
            mask = np.zeros((annotations['imageHeight'], annotations['imageWidth']), dtype=np.uint8)  # 创建空白掩码
            for shape in annotations['shapes']:
                points = np.array(shape['points'], dtype=np.int32)  # 将形状的点转换为数组
                cv2.fillPoly(mask, [points], color=(255))  # 用白色填充多边形
    else:
        raise ValueError("Unsupported label format. Use 'mask' or 'json'.")  # 抛出异常
    return mask  # 返回生成的掩码


def process_folder(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, label_format="mask",
                   n_augmentations=5, color_augmentation='pca'):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    image_files = os.listdir(input_image_folder)  # 获取输入图像文件列表

    for image_file in image_files:
        image_path = os.path.join(input_image_folder, image_file)  # 构建图像文件路径

        # 根据标签格式选择相应的掩码文件名
        if image_path.endswith('.jpg'):
            mask_file = image_file.replace('.jpg', '.png') if label_format == "mask" else image_file.replace(
                '.jpg', '.json')
        elif image_path.endswith('.bmp'):
            mask_file = image_file.replace('.bmp', '.png') if label_format == "mask" else image_file.replace(
                '.bmp', '.json')
        else:
            print(f"Unsupported image format for {image_path}, skipping.")
        mask_path = os.path.join(input_mask_folder, mask_file)  # 构建掩码文件路径
        if not os.path.exists(mask_path):
            print(f"Label file {mask_path} does not exist, skipping.")  # 如果掩码文件不存在，跳过该文件
            continue

        image = cv2.imread(image_path)  # 读取图像文件
        mask = read_mask(mask_path, label_format)  # 读取掩码文件

        augmented_images, augmented_masks = augment_image(image, mask, n_augmentations,
                                                          color_augmentation)  # 对图像和掩码进行数据增强

        for i, (image_aug, mask_aug) in enumerate(zip(augmented_images, augmented_masks)):
            output_image_path = os.path.join(output_image_folder,
                                             f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")  # 输出图像文件路径
            output_mask_path = os.path.join(output_mask_folder,
                                            f"{os.path.splitext(mask_file)[0]}_aug_{i}.png")  # 输出掩码文件路径


            cv2.imwrite(output_image_path, image_aug)  # 保存增强后的图像
            cv2.imwrite(output_mask_path, mask_aug)  # 保存增强后的掩码


if __name__ == "__main__":
    input_image_folder = r"F:\work\dataset\rebar2D\train2\img"
    input_mask_folder = r"F:\work\dataset\rebar2D\train2\mask"  # 输入掩码文件夹路径
    output_image_folder = r"F:\work\dataset\rebar2D\train2\img2"  # 输出图像文件夹路径
    output_mask_folder = r"F:\work\dataset\rebar2D\train2\mask2"  # 输出掩码文件夹路径

    label_format = "mask"  # 标签格式，可选"mask"或"json"
    n_augmentations =4  # 每张图像增强的数量
    color_augmentation = 'pca'  # 颜色增强方式，可选"pca"或"color"

    process_folder(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, label_format,
                   n_augmentations, color_augmentation)  # 处理文件夹中的文件
