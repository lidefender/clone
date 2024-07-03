import cv2
import numpy as np
import os
from ultralytics import YOLO

# 使用YOLOv8模型检测钢筋并分割
def detect_rebar(image_path, model):
    results = model(image_path)

    # 获取分割结果的边界框
    boxes = results[0].boxes.cpu().numpy()  # 假设返回的第一个结果包含边界框

    return boxes

# 获取边界框四个点并扩大
def expand_bbox(bbox, image_shape, expansion_factor=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    x1_exp = max(0, x1 - width * expansion_factor)
    y1_exp = max(0, y1 - height * expansion_factor)
    x2_exp = min(image_shape[1], x2 + width * expansion_factor)
    y2_exp = min(image_shape[0], y2 + height * expansion_factor)

    return int(x1_exp), int(y1_exp), int(x2_exp), int(y2_exp)

# 提取扩大后的边界框内容
def extract_expanded_bbox_content(image, bbox):
    x1, y1, x2, y2 = bbox
    imagenew = image[y1:y2, x1:x2]

    return imagenew

# 使用传统方法获得钢筋的轮廓
# def get_rebar_contour(image, min_aspect_ratio=2.0, min_area=500):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     # 腐蚀和膨胀操作
#     # kernel = np.ones((2,2),np.uint8)
#     # eroded = cv2.erode(blurred, kernel, iterations = 1)
#     # dilated = cv2.dilate(eroded, kernel, iterations = 1)
#
#     # _, thresh = cv2.threshold(dilated, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
#     # 找到轮廓
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     filtered_contours = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = float(w) / h
#         area = cv2.contourArea(contour)
#
#         if area >= min_area and aspect_ratio >= min_aspect_ratio:
#             filtered_contours.append(contour)
#
#     return filtered_contours

def get_rebar_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 将轮廓写入和原图同大小的画布中并保存
def draw_contours_on_canvas(image, contours, bbox, output_path):
    canvas = np.zeros_like(image)
    x1, y1, x2, y2 = bbox
    for contour in contours:
        contour += np.array([x1, y1])
        cv2.drawContours(canvas, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.imwrite(output_path, canvas)

# 处理文件夹中的所有图片
def process_folder(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
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
                print(f"No rebar detected in {filename}")
                continue

            # 假设只有一个钢筋对象，获取其边界框并扩大
            bbox = boxes[0].xyxy[0]
            expanded_bbox = expand_bbox(bbox, image.shape)

            # 提取扩大后的边界框内容
            expanded_bbox_content = extract_expanded_bbox_content(image, expanded_bbox)

            # 获取钢筋的轮廓
            contours = get_rebar_contour(expanded_bbox_content)

            # 将轮廓写入和原图同大小的画布中并保存
            draw_contours_on_canvas(image, contours, expanded_bbox, output_path)
            print(f'Result saved to {output_path}')



if __name__ == '__main__':
    model_choice = r"F:\work\python\clone\2d\ultralnew\ultralytics\best1.pt"
    input_folder = r"F:\work\dataset\rebar2D\train\img"  # 输入图像文件夹路径
    output_folder =r"F:\work\dataset\rebar2D\train\TEMP"  # 输出图像文件夹路径

    model = YOLO(model_choice)  # 根据输入的模型路径加载模型
    process_folder(input_folder, output_folder, model)
    print("欧了")
