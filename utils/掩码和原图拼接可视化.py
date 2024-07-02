from PIL import Image, ImageOps
import os

def apply_mask(original_image_path, mask_image_path, output_image_path):
    # 打开原图和掩码图像
    original_image = Image.open(original_image_path).convert("RGBA")
    mask_image = Image.open(mask_image_path).convert("RGBA")

    # 生成一个与掩码图像大小相同的透明图像
    transparent_image = Image.new("RGBA", mask_image.size, (0, 0, 0, 0))

    # 将掩码图像叠加到透明图像上
    transparent_image.paste(mask_image, (0, 0), mask_image)

    # 将生成的透明彩色掩码叠加到原图上
    combined_image = Image.alpha_composite(original_image, transparent_image)
    # if combined_image.mode == 'RGBA':
    #     combined_image = combined_image.convert('RGB')
    # 保存生成的图像
    combined_image.save(output_image_path)

def batch_process_images(original_folder, mask_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取原图和掩码图像文件名
    original_images = [f for f in os.listdir(original_folder) if f.endswith(('jpg', 'bmp'))]
    mask_images = [f for f in os.listdir(mask_folder) if f.endswith('png')]

    for original_image_name in original_images:
        # 对应的掩码图像文件名
        mask_image_name = original_image_name.rsplit('.', 1)[0] + '.png'

        if mask_image_name in mask_images:
            original_image_path = os.path.join(original_folder, original_image_name)
            mask_image_path = os.path.join(mask_folder, mask_image_name)
            output_image_path = os.path.join(output_folder, original_image_name.replace('.jpg', '.png'))

            # 应用掩码
            apply_mask(original_image_path, mask_image_path, output_image_path)
            print(f'Processed {original_image_name} with mask {mask_image_name}')
                # 在保存之前转换图像


if __name__ == "__main__":


    original_folder = r"F:\work\dataset\rebar2D\train\img"
    mask_folder = r"F:\work\dataset\rebar2D\train\mask"
    output_folder = r"F:\work\dataset\rebar2D\train\TEMP"

    batch_process_images(original_folder, mask_folder, output_folder)
