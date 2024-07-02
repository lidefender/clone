import cv2
import numpy as np

def apply_color_to_mask(mask, class_info):
    '''
    输入单通道灰度掩码图像，输出彩色掩码图像
    '''
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # 创建彩色空白图像

    for one_class in class_info:
        colored_mask[mask == one_class['label']] = one_class['color']

    return colored_mask

def draw_bounding_boxes(frame, mask, class_info):
    '''
    在图像上绘制边界框
    '''
    for one_class in class_info:
        class_label = one_class['label']
        contours, _ = cv2.findContours((mask == class_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 0:  # 忽略小的噪声
                x, y, w, h = cv2.boundingRect(contour)
                color = one_class['color']
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def main():
    # 假设你的模型已经加载并且可以生成预测的掩码
    # 这里用一个示例函数来模拟模型的预测
    def model_predict(frame):
        # 示例：生成一个随机的单通道掩码
        mask = np.random.randint(0, 3, frame.shape[:2], dtype=np.uint8)
        return mask

    class_info = [
        {'label': 1, 'color': (0, 255, 0)},  # 绿色
        {'label': 2, 'color': (255, 0, 0)},  # 红色
        # 可以根据需要添加更多类别
    ]

    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    show_bboxes = True  # 是否显示边界框

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = model_predict(frame)  # 获取模型预测的掩码
        colored_mask = apply_color_to_mask(mask, class_info)  # 将灰度掩码转换为彩色掩码

        # 将彩色掩码与原图进行叠加
        overlay_img = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        if show_bboxes:
            draw_bounding_boxes(overlay_img, mask, class_info)  # 绘制边界框

        cv2.imshow('Overlay', overlay_img)  # 显示叠加图像

        key = cv2.waitKey(1)
        if key == ord('q'):  # 按 'q' 键退出
            break
        elif key == ord('b'):  # 按 'b' 键切换边界框显示
            show_bboxes = not show_bboxes

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
