import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import albumentations as A  # 导入Albumentations库，用于图像增强
import os  # 导入OS库，用于文件和目录操作
import json  # 导入JSON库，用于处理JSON格式的数据
from tqdm import tqdm


def pca_color_augmentation(image, alpha_std=0.07):
    """ 使用PCA对图像进行颜色增强 """
    orig_img = image.astype(float)  # 将图像转换为浮点数类型
    img_rs = orig_img.reshape(-1, 3)  # 将图像重塑为二维数组
    img_mean = np.mean(img_rs, axis=0)  # 计算每个通道的均值
    img_std = np.std(img_rs, axis=0)  # 计算每个通道的标准差
    img_rs = (img_rs - img_mean) / img_std  # 对图像进行标准化处理

    cov = np.cov(img_rs, rowvar=False)  # 计算协方差矩阵
    eigvals, eigvecs = np.linalg.eigh(cov)  # 计算特征值和特征向量

    noise = np.random.normal(0, alpha_std, 3)  # 生成正态分布的噪声
    noise = eigvecs @ (eigvals ** 0.2) * noise  # 应用PCA变换
    noise = (noise * img_std) + img_mean * 0.08  # 反标准化处理

    aug_img = orig_img + noise  # 将噪声加到原图像上
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)  # 裁剪值并转换为8位无符号整数
    return aug_img  # 返回增强后的图像


def is_image_too_bright_or_dark(image, lower_threshold=30, upper_threshold=220):
    """ 检查图像是否过曝或太黑 """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_image)
    if mean_brightness < lower_threshold:
        print(f"Image is too dark: {mean_brightness}")
    elif mean_brightness > upper_threshold:
        print(f"Image is too bright: {mean_brightness}")
    return mean_brightness < lower_threshold or mean_brightness > upper_threshold


def augment_image(image, mask, n_augmentations=1):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转，概率为50%
        A.VerticalFlip(p=0.5),  # 垂直翻转，概率为50%
        A.RandomRotate90(p=0.5),  # 随机旋转90度，概率为50%
        A.Transpose(p=0.5),  # 转置操作（交换x和y轴），概率为50%
        A.RandomBrightnessContrast(p=0.2),  # 随机调整亮度和对比度，概率为20%
        A.ElasticTransform(p=0.2),  # 弹性变换，概率为20%
        A.GridDistortion(p=0.2),  # 网格扭曲，概率为20%
        A.ShiftScaleRotate(shift_limit=0.0825, scale_limit=0.1, rotate_limit=45, p=0.3),  # 平移、缩放和旋转，概率为30%
        A.RandomCrop(height=1080, width=1920, p=0.2),  # 随机裁剪到640x640大小，概率为40%
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # 高斯噪声，概率为20%
        A.MotionBlur(blur_limit=3, p=0.2),  # 模糊，概率为20%
    ])

    augmented_images = []
    augmented_masks = []

    for _ in range(n_augmentations):
        while True:
            augmented = transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']
            image_aug = pca_color_augmentation(image_aug)

            if not is_image_too_bright_or_dark(image_aug):
                break

        augmented_images.append(image_aug)
        augmented_masks.append(mask_aug)

    return augmented_images, augmented_masks


def read_mask(mask_path, label_format):
    if label_format == "mask":
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    elif label_format == "json":
        with open(mask_path, 'r') as f:
            annotations = json.load(f)
            mask = np.zeros((annotations['imageHeight'], annotations['imageWidth']), dtype=np.uint8)
            for shape in annotations['shapes']:
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], color=(255))
    else:
        raise ValueError("Unsupported label format. Use 'mask' or 'json'.")
    return mask


def process_folder(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, label_format="mask",
                   n_augmentations=5):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    image_files = os.listdir(input_image_folder)

    for image_file in tqdm(image_files):
        image_path = os.path.join(input_image_folder, image_file)

        if image_path.endswith('.jpg'):
            mask_file = image_file.replace('.jpg', '.png') if label_format == "mask" else image_file.replace('.jpg',
                                                                                                             '.json')
        elif image_path.endswith('.bmp'):
            mask_file = image_file.replace('.bmp', '.png') if label_format == "mask" else image_file.replace('.bmp',
                                                                                                             '.json')
        else:
            print(f"Unsupported image format for {image_path}, skipping.")
            continue

        mask_path = os.path.join(input_mask_folder, mask_file)

        if not os.path.exists(mask_path):
            print(f"Label file {mask_path} does not exist, skipping.")
            continue

        image = cv2.imread(image_path)
        mask = read_mask(mask_path, label_format)

        augmented_images, augmented_masks = augment_image(image, mask, n_augmentations)

        for i, (image_aug, mask_aug) in enumerate(zip(augmented_images, augmented_masks)): # zip函数用于将augmented_images和augmented_masks两个列表的对应元素打包成一个个元组，然后返回这些元组组成的迭代器。
            output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")
            output_mask_path = os.path.join(output_mask_folder, f"{os.path.splitext(mask_file)[0]}_aug_{i}.png")

            cv2.imwrite(output_image_path, image_aug)
            cv2.imwrite(output_mask_path, mask_aug)


if __name__ == "__main__":
    input_image_folder = r"F:\work\dataset\rebar2D\train2\img"
    input_mask_folder = r"F:\work\dataset\rebar2D\train2\mask"
    output_image_folder = r"F:\work\dataset\rebar2D\train2\img3"
    output_mask_folder = r"F:\work\dataset\rebar2D\train2\mask3"

    label_format = "mask"
    n_augmentations = 3

    process_folder(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, label_format,
                   n_augmentations)
