import cv2
import numpy as np
import os
from ultralytics import YOLO


# 使用YOLOv8模型检测钢筋并分割
def detect_rebar(image_path, model):
    # 使用模型对图像进行检测
    results = model(image_path)

    # 获取分割结果的边界框
    boxes = results[0].boxes.cpu().numpy()  # 假设返回的第一个结果包含边界框

    return boxes


# 获取边界框四个点并扩大
def expand_bbox(bbox, image_shape, expansion_factor=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # 计算扩大的边界框坐标
    x1_exp = max(0, x1 - width * expansion_factor)
    y1_exp = max(0, y1 - height * expansion_factor)
    x2_exp = min(image_shape[1], x2 + width * expansion_factor)
    y2_exp = min(image_shape[0], y2 + height * expansion_factor)

    return int(x1_exp), int(y1_exp), int(x2_exp), int(y2_exp)


# 提取扩大后的边界框内容
def extract_expanded_bbox_content(image, bbox):
    x1, y1, x2, y2 = bbox
    # 提取图像中边界框内的内容
    imagenew = image[y1:y2, x1:x2]

    return imagenew


# 将抠出的内容放到与原图同样大小的画布中
def place_on_canvas(image, content, bbox, save_as_png=False):
    if save_as_png:
        # 创建一个透明的画布
        canvas = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    else:
        # 创建一个黑色的画布
        canvas = np.zeros_like(image)

    x1, y1, x2, y2 = bbox
    if save_as_png:
        # 将内容放置到画布上
        canvas[y1:y2, x1:x2, :3] = content
        canvas[y1:y2, x1:x2, 3] = 255  # 设置透明度通道为不透明
    else:
        # 将内容放置到画布上
        canvas[y1:y2, x1:x2] = content

    return canvas


# 处理文件夹中的所有图片
def process_folder(input_folder, output_folder, model, save_as_png=False):
    if not os.path.exists(output_folder):
        # 如果输出文件夹不存在，则创建
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图像
            image = cv2.imread(image_path)

            # 检测钢筋并获取边界框
            boxes = detect_rebar(image_path, model)

            if len(boxes) == 0:
                # 如果没有检测到钢筋，则跳过该图像
                print(f"No rebar detected in {filename}")
                continue

            # 假设只有一个钢筋对象，获取其边界框并扩大
            bbox = boxes[0].xyxy[0]
            expanded_bbox = expand_bbox(bbox, image.shape)

            # 提取扩大后的边界框内容
            expanded_bbox_content = extract_expanded_bbox_content(image, expanded_bbox)

            # 将抠出的内容放到与原图同样大小的画布中
            canvas = place_on_canvas(image, expanded_bbox_content, expanded_bbox, save_as_png)

            if save_as_png:
                # 如果选择保存为PNG格式，则修改输出路径的扩展名并保存
                output_path = os.path.splitext(output_path)[0] + ".png"
                cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                # 否则保存为JPG格式
                cv2.imwrite(output_path, canvas)

            print(f'Result saved to {output_path}')


if __name__ == '__main__':
    model_choice = r"F:\work\python\clone\2d\ultralnew\ultralytics\best1.pt"  # 模型路径
    input_folder = r"F:\work\dataset\rebar2D\train\img"  # 输入图像文件夹路径
    output_folder = r"F:\work\dataset\rebar2D\train\TEMP"  # 输出图像文件夹路径

    save_as_png = True  # 设置为True则保存为PNG格式，默认False保存为JPG格式

    # 根据输入的模型路径加载模型
    model = YOLO(model_choice)
    # 处理文件夹中的所有图片
    process_folder(input_folder, output_folder, model, save_as_png)
    print("欧了")