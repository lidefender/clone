import os
import random
import shutil
from pathlib import Path
from typing import List


def split_dataset(
        image_dir: str,
        label_dir: str,
        train_image_dir: str,
        val_image_dir: str,
        train_label_dir: str,
        val_label_dir: str,
        val_ratio: float = 0.2,
        seed: int = 42
) -> None:
    """
    自动分离训练集和验证集的函数，带有细分的图像和标签目录。

    :param image_dir: 包含图像文件的目录
    :param label_dir: 包含标签文件的目录
    :param train_image_dir: 保存训练集图像的目录
    :param val_image_dir: 保存验证集图像的目录
    :param train_label_dir: 保存训练集标签的目录
    :param val_label_dir: 保存验证集标签的目录
    :param val_ratio: 验证集所占比例，默认值为0.2
    :param seed: 随机种子，默认值为42
    """
    random.seed(seed)

    # 创建训练集和验证集目录
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 打乱图像文件列表
    random.shuffle(image_files)

    # 计算验证集大小
    val_size = int(len(image_files) * val_ratio)

    # 分割数据集
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]

    def copy_files(files: List[str], src_image_dir: str, src_label_dir: str, dst_image_dir: str,
                   dst_label_dir: str) -> None:
        """
        复制图像及其标签文件到目标目录。

        :param files: 要复制的文件列表
        :param src_image_dir: 源图像目录
        :param src_label_dir: 源标签目录
        :param dst_image_dir: 目标图像目录
        :param dst_label_dir: 目标标签目录
        """
        for file_name in files:
            # 复制图像文件
            shutil.copy(os.path.join(src_image_dir, file_name), os.path.join(dst_image_dir, file_name))

            # 复制标签文件
            base_name = os.path.splitext(file_name)[0]
            for ext in ['.json', '.png', '.txt']:
                label_file = os.path.join(src_label_dir, base_name + ext)
                if os.path.exists(label_file):
                    shutil.copy(label_file, os.path.join(dst_label_dir, base_name + ext))

    # 复制验证集文件
    copy_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)

    # 复制训练集文件
    copy_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)

    print(f"训练集大小: {len(train_files)}")
    print(f"验证集大小: {len(val_files)}")


# 示例用法
image_directory = r"F:\work\dataset\rebar2D\train2\img2"
label_directory = r"F:\work\dataset\rebar2D\train2\yolotxt"
train_image_directory = r"F:\work\dataset\rebar2D\yolodataset\train"
val_image_directory = r"F:\work\dataset\rebar2D\yolodataset\val"
train_label_directory = r"F:\work\dataset\rebar2D\yolodataset\train"
val_label_directory = r"F:\work\dataset\rebar2D\yolodataset\val"

split_dataset(image_directory, label_directory, train_image_directory, val_image_directory, train_label_directory,
              val_label_directory, val_ratio=0.08)


