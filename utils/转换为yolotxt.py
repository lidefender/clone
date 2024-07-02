import os  # 导入操作系统模块
import json  # 导入JSON模块


def convert_json_to_txt_with_normalization_and_mapping(labelme_dir, output_dir, label_mapping):
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录，如果不存在则创建

    json_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]  # 获取目录下所有JSON文件的列表

    for filename in json_files:  # 遍历每个JSON文件
        json_path = os.path.join(labelme_dir, filename)  # 构建JSON文件的完整路径
        txt_filename = filename.replace('.json', '.txt')  # 构建输出TXT文件的文件名
        txt_path = os.path.join(output_dir, txt_filename)  # 构建输出TXT文件的完整路径

        with open(json_path) as f:  # 打开JSON文件
            data = json.load(f)  # 加载JSON数据

        image_height = data['imageHeight']  # 获取图像的高度
        image_width = data['imageWidth']  # 获取图像的宽度

        with open(txt_path, 'w') as f:  # 打开（或创建）TXT文件以写入模式
            for shape in data['shapes']:  # 遍历每个形状
                label = shape['label']  # 获取形状的标签
                points = shape['points']  # 获取形状的坐标点
                normalized_points = [(x / image_width, y / image_height) for x, y in points]  # 归一化坐标点
                mapped_label = label_mapping.get(label, -1)  # 获取标签的映射值，默认为-1

                # 将标签和所有归一化后的坐标点写入TXT文件，一行表示一个实例
                points_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])  # 将坐标点转换为字符串
                f.write(f"{mapped_label} {points_str}\n")  # 写入TXT文件


# 使用示例


labelme_dir = r"F:\work\dataset\rebar2D\train2\label2"  # 指定Labelme JSON文件所在目录
output_dir = r"F:\work\dataset\rebar2D\train2\yolotxt"  # 指定输出的TXT文件目录
label_mapping = {  # 标签映射字典
    "rebar": 0,
    "socket": 1
    # 添加更多标签映射
}
convert_json_to_txt_with_normalization_and_mapping(labelme_dir, output_dir, label_mapping)  # 调用函数进行转换
