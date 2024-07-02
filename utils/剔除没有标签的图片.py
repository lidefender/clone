import os
import shutil

# 定义文件夹路径
image_folder = r"F:\work\dataset\rebar2D\train\img2"
json_folder = r"F:\work\dataset\rebar2D\train\img2"
destination_folder = r"F:\work\dataset\rebar2D\train\imgnolabel"

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 遍历图片文件夹
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):  # 根据需要修改图片文件的后缀
        image_name = os.path.splitext(image_file)[0]
        json_file = f"{image_name}.json"

        # 检查是否有对应的JSON文件
        if not os.path.exists(os.path.join(json_folder, json_file)):
            # 没有对应的JSON文件，移动图片到目标文件夹
            shutil.move(os.path.join(image_folder, image_file), os.path.join(destination_folder, image_file))
            print(f"Moved {image_file} to {destination_folder}")
