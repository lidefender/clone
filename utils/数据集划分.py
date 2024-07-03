import os
import shutil
from sklearn.model_selection import train_test_split


# 定义函数以创建文件夹
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 定义数据集划分函数
def split_dataset(image_dir, label_dir, output_dir, test_size=0.2, val_size=0.1, test_split=False):
    # 创建输出目录
    make_dir(output_dir)
    make_dir(os.path.join(output_dir, 'train'))
    make_dir(os.path.join(output_dir, 'val'))

    if test_split:
        make_dir(os.path.join(output_dir, 'test'))

    # 获取所有图像文件名
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    # 将文件名排序，以确保图像和标签对齐
    images.sort()
    labels.sort()

    # 划分训练集和临时集（包含验证集和可能的测试集）
    train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels,
                                                                            test_size=(val_size + test_size))

    # 划分验证集和测试集
    if test_split:
        val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=(
                    test_size / (val_size + test_size)))
    else:
        val_images, val_labels = temp_images, temp_labels

    # 定义复制文件的函数
    def copy_files(file_list, source_dir, dest_dir):
        for file in file_list:
            shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))

    # 复制训练集文件
    copy_files(train_images, image_dir, os.path.join(output_dir, 'train'))
    copy_files(train_labels, label_dir, os.path.join(output_dir, 'train'))

    # 复制验证集文件
    copy_files(val_images, image_dir, os.path.join(output_dir, 'val'))
    copy_files(val_labels, label_dir, os.path.join(output_dir, 'val'))

    # 如果需要，复制测试集文件
    if test_split:
        copy_files(test_images, image_dir, os.path.join(output_dir, 'test'))
        copy_files(test_labels, label_dir, os.path.join(output_dir, 'test'))


# 示例使用
# input_dir = r"F:\work\python\clone\dataset\rebar2d\images"
# output_dir = r"F:\work\python\clone\2d\ultralytics\dataset\rebar2d"
image_dir = r"F:\work\python\clone\dataset\rebar2d\images"  # 图像目录路径
label_dir = r'F:\work\python\clone\dataset\rebar2d\yololabel'  # 标签目录路径
output_dir = r"F:\work\python\clone\2d\ultralytics\dataset\rebar2d"  # 输出目录路径

split_dataset(image_dir, label_dir, output_dir, test_size=0.1, val_size=0, test_split=False)
