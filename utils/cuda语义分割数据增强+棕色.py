import cv2
import numpy as np
import torch
import albumentations as A
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")  # 输出设备信息

def pca_color_augmentation(image, alpha_std=0.1):
    """ 使用PCA对图像进行颜色增强 """
    orig_img = torch.tensor(image.astype(np.float32), device=device)  # 将图像转换为浮点数类型并移动到GPU
    img_rs = orig_img.view(-1, 3)  # 将图像重塑为二维数组
    img_mean = torch.mean(img_rs, dim=0)  # 计算每个通道的均值
    img_std = torch.std(img_rs, dim=0)  # 计算每个通道的标准差
    img_rs = (img_rs - img_mean) / img_std  # 对图像进行标准化处理

    cov = torch.mm(img_rs.t(), img_rs) / img_rs.size(0)  # 计算协方差矩阵
    eigvals, eigvecs = torch.linalg.eigh(cov)  # 计算特征值和特征向量，使用eigh代替eig以确保实数特征值

    noise = torch.normal(mean=0, std=alpha_std, size=(3,), device=device)  # 生成正态分布的噪声
    noise = torch.mv(eigvecs, eigvals.sqrt() * noise)  # 应用PCA变换
    noise = (noise * img_std) + img_mean  # 反标准化处理

    aug_img = orig_img + noise.unsqueeze(0)  # 将噪声加到原图像上，并确保噪声形状与原图像相同
    aug_img = torch.clamp(aug_img, 0, 255).byte()  # 裁剪值并转换为8位无符号整数
    return aug_img.cpu().numpy()  # 将图像移回CPU并返回
def color_augment(image, alpha=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间

    lower_black_iron = np.array([0, 0, 0])  # 黑铁色的下限
    upper_black_iron = np.array([180, 255, 70])  # 黑铁色的上限

    mask = cv2.inRange(hsv, lower_black_iron, upper_black_iron)  # 创建黑铁色区域的掩码

    silver_tint = torch.tensor([192, 192, 192], dtype=torch.float32, device=device)

    image = torch.tensor(image.astype(np.float32), device=device)  # 将图像转换为浮点数格式并移动到GPU
    mask = torch.tensor(mask, device=device)  # 移动掩码到GPU

    image[mask > 0] = (1 - alpha) * image[mask > 0] + alpha * silver_tint

    image[mask > 0] = torch.clamp(image[mask > 0] * 1.5, 0, 255).byte()  # 增加亮度和对比度

    return image.cpu().numpy()  # 将图像移回CPU并返回

def augment_image(image, mask, n_augmentations=1, color_augmentation='pca'):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转，概率为50%
        A.VerticalFlip(p=0.5),  # 垂直翻转，概率为50%
        A.RandomRotate90(p=0.5),  # 随机旋转90度，概率为50%
        A.RandomBrightnessContrast(p=0.2),  # 随机调整亮度和对比度，概率为20%
        A.ElasticTransform(p=0.2),  # 弹性变换，概率为20%
        A.GridDistortion(p=0.2),  # 网格扭曲，概率为20%
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),  # 平移、缩放和旋转，概率为20%
    ])

    augmented_images = []  # 存储增强后的图像
    augmented_masks = []  # 存储增强后的掩码

    for _ in range(n_augmentations):
        augmented = transform(image=image, mask=mask)  # 对图像和掩码进行增强
        image_aug = augmented['image']  # 获取增强后的图像
        mask_aug = augmented['mask']  # 获取增强后的掩码

        if color_augmentation == 'pca':
            image_aug = pca_color_augmentation(image_aug)
        elif color_augmentation == 'color':
            image_aug = color_augment(image_aug)

        augmented_images.append(image_aug)  # 添加到增强图像列表
        augmented_masks.append(mask_aug)  # 添加到增强掩码列表

    return augmented_images, augmented_masks  # 返回增强后的图像和掩码

def read_mask(mask_path, label_format):
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
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    image_files = os.listdir(input_image_folder)  # 获取输入图像文件列表

    for image_file in image_files:
        image_path = os.path.join(input_image_folder, image_file)  # 构建图像文件路径

        if image_path.endswith('.jpg'):
            mask_file = image_file.replace('.jpg', '.png') if label_format == "mask" else image_file.replace(
                '.jpg', '.json')
        elif image_path.endswith('.bmp'):
            mask_file = image_file.replace('.bmp', '.png') if label_format == "mask" else image_file.replace(
                '.bmp', '.json')
        else:
            print(f"Unsupported image format for {image_path}, skipping.")
            continue

        mask_path = os.path.join(input_mask_folder, mask_file)  # 构建掩码文件路径
        if not os.path.exists(mask_path):
            print(f"Label file {mask_path} does not exist, skipping.")  # 如果掩码文件不存在，跳过该文件
            continue

        image = cv2.imread(image_path)  # 读取图像文件
        mask = read_mask(mask_path, label_format)  # 读取掩码文件

        augmented_images, augmented_masks = augment_image(image, mask, n_augmentations, color_augmentation)  # 对图像和掩码进行数据增强

        for i, (image_aug, mask_aug) in enumerate(zip(augmented_images, augmented_masks)):
            output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")  # 输出图像文件路径
            output_mask_path = os.path.join(output_mask_folder, f"{os.path.splitext(mask_file)[0]}_aug_{i}.png")  # 输出掩码文件路径

            cv2.imwrite(output_image_path, image_aug)  # 保存增强后的图像
            cv2.imwrite(output_mask_path, mask_aug)  # 保存增强后的掩码

if __name__ == "__main__":
    input_image_folder = r"F:\work\dataset\rebar2D\train2\img"
    input_mask_folder = r"F:\work\dataset\rebar2D\train2\mask"  # 输入掩码文件夹路径
    output_image_folder = r"F:\work\dataset\rebar2D\train2\img2"  # 输出图像文件夹路径
    output_mask_folder = r"F:\work\dataset\rebar2D\train2\mask2"  # 输出掩码文件夹路径

    label_format = "mask"  # 标签格式，可选"mask"或"json"
    n_augmentations = 3  # 每张图像增强的数量
    color_augmentation = 'pca'  # 颜色增强方式，可选"pca"或"
    process_folder(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, label_format, n_augmentations, color_augmentation)