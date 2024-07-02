from ultralytics import YOLO
import torch.nn as nn

# 定义新的卷积层
class NewConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NewConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 例如，加载yolov8n的预训练权重

# 打印模型结构以查看层次信息
print(model.model)

# 获取模型的第一层信息
first_conv_layer = model.model[0]

# 创建新的卷积层，保持输入输出通道数与原始层一致
new_layer = NewConvLayer(first_conv_layer.conv.in_channels,
                         first_conv_layer.conv.out_channels,
                         first_conv_layer.conv.kernel_size,
                         first_conv_layer.conv.stride,
                         first_conv_layer.conv.padding)

# 替换模型的第一层
model.model[0] = new_layer

# 打印模型结构以验证替换情况
print(model.model)

# 定义你的训练数据集和相关设置
# 比如 data = 'data/coco.yaml'
# model.train(data=data)
