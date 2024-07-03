import cv2
import numpy as np
import albumentations as A
import os
import json

def read_json_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def create_mask_from_annotations(annotations):
    mask = np.zeros((annotations['imageHeight'], annotations['imageWidth']), dtype=np.uint8)
    for shape in annotations['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=(255))
    return mask

def augment_image_and_annotations(image, annotations, n_augmentations=1):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0825, scale_limit=0.1, rotate_limit=45, p=0.3),
        A.RandomCrop(height=640, width=640, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
    ], keypoint_params=A.KeypointParams(format='xy'))

    mask = create_mask_from_annotations(annotations)
    augmented_images = []
    augmented_annotations = []

    for _ in range(n_augmentations):
        augmented = transform(image=image, mask=mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']

        aug_annotations = {
            'imageHeight': mask_aug.shape[0],
            'imageWidth': mask_aug.shape[1],
            'shapes': []
        }

        for shape in annotations['shapes']:
            points = np.array(shape['points'], dtype=np.float32)
            transformed_points = A.ReplayCompose.replay(augmented['replay'], image=np.zeros(mask_aug.shape, dtype=np.uint8), mask=mask, keypoints=[tuple(point) for point in points])[2]['keypoints']
            aug_annotations['shapes'].append({
                'label': shape['label'],
                'points': transformed_points
            })

        augmented_images.append(image_aug)
        augmented_annotations.append(aug_annotations)

    return augmented_images, augmented_annotations

def process_folder(input_image_folder, input_json_folder, output_image_folder, output_json_folder, n_augmentations=5):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)

    image_files = os.listdir(input_image_folder)

    for image_file in image_files:
        image_path = os.path.join(input_image_folder, image_file)
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(input_json_folder, json_file)

        if not os.path.exists(json_path):
            print(f"JSON file {json_path} does not exist, skipping.")
            continue

        image = cv2.imread(image_path)
        annotations = read_json_annotations(json_path)

        augmented_images, augmented_annotations = augment_image_and_annotations(image, annotations, n_augmentations)

        for i, (image_aug, annotation_aug) in enumerate(zip(augmented_images, augmented_annotations)):
            output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")
            output_json_path = os.path.join(output_json_folder, f"{os.path.splitext(json_file)[0]}_aug_{i}.json")

            cv2.imwrite(output_image_path, image_aug)
            with open(output_json_path, 'w') as f:
                json.dump(annotation_aug, f, indent=4)

if __name__ == "__main__":
    input_image_folder = r"F:\work\dataset\rebar2D\train2\img"
    input_json_folder = r"F:\work\dataset\rebar2D\train2\label"
    output_image_folder = r"F:\work\dataset\rebar2D\train2\img3"
    output_json_folder = r"F:\work\dataset\rebar2D\train2\label3"

    n_augmentations = 4

    process_folder(input_image_folder, input_json_folder, output_image_folder, output_json_folder, n_augmentations)
