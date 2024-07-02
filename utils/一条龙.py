from 用感兴趣区域方法得到抠图 import *
from ultralytics.data.annotator import auto_annotate
from yolotxt转json import *

input_folder = r"F:\work\dataset\rebar2D\train\img2"  # 输入图像文件夹路径
ROIoutput_folder = r"F:\work\dataset\rebar2D\train\TEMP"  # 输出图像文件夹路径

label_folder = r"F:\work\dataset\rebar2D\train\TEMP_auto_annotate_labels"
image_folder = input_folder
# json_output_folder = r"F:\work\dataset\rebar2D\train\jsontest2"
json_output_folder = input_folder
# 定义标签映射
label_map = {
    0: 'rebar',

    # 添加更多标签映射
}

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

# 显示图像并设置鼠标回调函数
cv2.namedWindow("image")
cv2.setMouseCallback("image", select_roi)
cv2.imshow("image", img_resized)
cv2.waitKey(0)

if len(roi_pts) == 2:
    # 处理文件夹中的所有图片
    process_folder(input_folder, ROIoutput_folder, roi_pts)
    print("处理完成")
else:
    print("Error: ROI not selected properly.")
cv2.destroyAllWindows()



print("开始自动标注")
auto_annotate(data=r"F:\work\dataset\rebar2D\train\TEMP", det_model=r"F:\work\python\clone\2d\ultralnew\ultralytics\best1.pt", sam_model=r'F:\work\python\clone\utils\mask\sam_b.pt')

print("开始转json")

batch_process(label_folder, image_folder, json_output_folder, label_map)
print("欧了")

