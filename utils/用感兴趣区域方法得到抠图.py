import cv2
import numpy as np
import os

# 鼠标回调函数，用于获取ROI
def select_roi(event, x, y, flags, param):
    global roi_pts, selecting, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(int(x / scale), int(y / scale))]
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        img_copy = img_resized.copy()
        cv2.rectangle(img_copy, (int(roi_pts[0][0] * scale), int(roi_pts[0][1] * scale)), (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts.append((int(x / scale), int(y / scale)))
        selecting = False
        cv2.rectangle(img_resized, (int(roi_pts[0][0] * scale), int(roi_pts[0][1] * scale)), (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img_resized)

# 提取ROI内容
def extract_roi_content(image, roi):
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    return image[y1:y2, x1:x2]

# 将抠出的内容放到与原图同样大小的透明画布中
def place_on_canvas(image, content, roi):
    canvas = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    canvas[y1:y2, x1:x2, :3] = content
    canvas[y1:y2, x1:x2, 3] = 255  # 设置透明度通道为不透明
    return canvas

# 处理文件夹中的所有图片
def process_folder(input_folder, output_folder, roi):
    if not os.path.exists(output_folder):
        # 如果输出文件夹不存在，则创建
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not open or find the image {filename}.")
                continue

            # 提取ROI内容
            roi_content = extract_roi_content(image, roi)

            # 将抠出的内容放到与原图同样大小的透明画布中
            canvas = place_on_canvas(image, roi_content, roi)

            # 保存结果为PNG格式
            cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            print(f'Result saved to {output_path}')

if __name__ == '__main__':
    input_folder = r"F:\work\dataset\rebar2D\train\img2"  # 输入图像文件夹路径
    output_folder = r"F:\work\dataset\rebar2D\train\TEMP"  # 输出图像文件夹路径

    # 获取文件夹中的第一张图像
    first_image_path = None
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            first_image_path = os.path.join(input_folder, filename)
            break

    if first_image_path is None:
        print("Error: No images found in the input folder.")
        exit()

    # 读取第一张图像并检查是否成功读取
    img = cv2.imread(first_image_path)
    if img is None:
        print("Error: Could not open or find the first image.")
        exit()

    # 缩放图像以适应窗口
    max_width = 800
    scale = max_width / img.shape[1]
    img_resized = cv2.resize(img, (max_width, int(img.shape[0] * scale)))

    roi_pts = []
    selecting = False

    # 显示图像并设置鼠标回调函数
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_roi)
    cv2.imshow("image", img_resized)
    cv2.waitKey(0)

    if len(roi_pts) == 2:
        # 处理文件夹中的所有图片
        process_folder(input_folder, output_folder, roi_pts)
        print("处理完成")
    else:
        print("Error: ROI not selected properly.")

    cv2.destroyAllWindows()