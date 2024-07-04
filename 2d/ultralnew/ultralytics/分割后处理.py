from ultralytics import YOLO
import cv2
import os
import numpy as np
from pycrfsuite import crf

model = YOLO(r"F:\warehouse\download\best (1).pt")  # load a pretrained model (recommended for training)

results = model(r"F:\work\dataset\rebar2D\train\img2\Image_20240701174633073.jpg")
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen

    result.save(filename="result2.jpg")  # save to disk

    # model = YOLO("yolov8x-seg.yaml").load("yolov10x.pt")  # build from YAML and transfer weights
    # Load a model
    # model = YOLO("yolov8x-seg.yaml")  # build a new model from YAML
    # Train the model
    # results = model.train(data="dataset/rebar2d.yaml",epochs=100, imgsz=640)



    # 定义函数来应用形态学滤波
    def apply_morphological_filtering(mask):
        # 腐蚀操作去除细小区域
        eroded_mask = cv2.erode(mask, kernel=np.ones((3, 3), dtype=np.uint8))
        # 膨胀操作填充空洞
        dilated_mask = cv2.dilate(eroded_mask, kernel=np.ones((3, 3), dtype=np.uint8))
        return dilated_mask


    # 定义函数来应用条件随机场 (CRF)
    def apply_crf(image, mask, labels):
        # 创建 CRF 对象
        crf = crf.CRF(n_labels=len(labels))

        # 添加特征
        crf.add_potentials(mask.flatten(), weight=1.0)
        crf.add_pairwise_potentials(image, labels)

        # 进行推断
        pred = crf.predict(mask.flatten())

        # 将预测结果转换为掩码
        pred_mask = np.reshape(pred, mask.shape)
        return pred_mask


    # 假设您已经准备好预测的 bounding box 和 mask，并将其存储在以下变量中：
    bounding_box = [x_min, y_min, x_max, y_max]
    mask = cv2.imread('mask.png')  # 假设 mask.png 是预测的掩码图像

    # 将 bounding box 应用于掩码
    mask = mask[y_min:y_max, x_min:x_max]

    # 应用形态学滤波
    filtered_mask = apply_morphological_filtering(mask)

    # 定义 CRF 标签
    labels = [0, 1]  # 0 表示背景，1 表示前景

    # 应用条件随机场 (CRF)
    refined_mask = apply_crf(mask, filtered_mask, labels)

    # 可视化结果
    cv2.imshow('原始掩码', mask)
    cv2.imshow('形态学滤波后掩码', filtered_mask)
    cv2.imshow('CRF 平滑后掩码', refined_mask)
    cv2.waitKey(0)
