from PIL import Image
import numpy as np
import os


def get_max_gray_value(image_path):
    # 打开图像
    img = Image.open(image_path).convert('L')  # 转换为灰度图像

    # 转换为numpy数组
    img_array = np.array(img)

    # 获取最高灰度值
    max_gray_value = img_array.max()
    print(f"最高灰度值: {max_gray_value}")

    return img_array, img


def replace_gray_value(img_array, old_value, new_value):
    # 将灰度值为old_value的像素转换为new_value
    img_array[img_array == old_value] = new_value

    # 转换为图像
    new_img = Image.fromarray(img_array)

    return new_img


def intowrith(mask_path):


            # 获取最高灰度值并返回图像数组
    img_array, img = get_max_gray_value(mask_path)

    # 将灰度值为1的像素转换为255
    new_img = replace_gray_value(img_array, 1, 255)
    return new_img
    # 保存新图像



if __name__ == "__main__":
    # 替换为您的图片路径
    # image_path = 'path_to_your_image.png'
    image_path = r"F:\work\dataset\rebar2D\train\mask1"
    output_dir = r"F:\work\dataset\rebar2D\train\TEMP"
    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            input_path = os.path.join(image_path, filename)
            # 获取最高灰度值并返回图像数组
            img_array, img = get_max_gray_value(input_path)

            # 将灰度值为1的像素转换为255
            new_img = replace_gray_value(img_array, 1, 255)

            # 保存新图像
            new_image_path = os.path.join(output_dir, filename)
            new_img.save(new_image_path)
            print(f"新图像已保存到: {new_image_path}")
    # # 获取最高灰度值并返回图像数组
    # img_array, img = get_max_gray_value("/kaggle/input/rebar2d/train2/mask/Image_20240622152006003_mask.png")
    #
    # # 将灰度值为1的像素转换为255
    # new_img = replace_gray_value(img_array, 1, 255)
    #
    # # 保存新图像
    # new_image_path = 'path_to_save_new_image.png'
    # new_img.save(new_image_path)
    # print(f"新图像已保存到: {new_image_path}")
