import sys
import cv2
import numpy as np
from ctypes import *

# 添加 MVS SDK 的 Python 包路径
sys.path.append(r"F:\install\MVS\Development\Samples\Python\MvImport")

# 导入 MVS SDK
from MvCameraControl_class import *

def main():
    # 创建相机对象
    cam = MvCamera()

    # 设备信息列表
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # 枚举设备
    ret = cam.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print(f"Enum devices failed! ret[0x{ret:x}]")
        return

    if deviceList.nDeviceNum == 0:
        print("No devices found!")
        return

    # 选择第一个设备
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    # 创建句柄
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"Create handle failed! ret[0x{ret:x}]")
        return

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"Open device failed! ret[0x{ret:x}]")
        return

    print("Device opened successfully")

    # 获取并打印当前分辨率
    width = MVCC_INTVALUE()  # 定义获取宽度的对象
    height = MVCC_INTVALUE()  # 定义获取高度的对象
    ret = cam.MV_CC_GetIntValue("Width", width)  # 获取宽度值
    if ret != 0:
        print(f"Get width failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return
    ret = cam.MV_CC_GetIntValue("Height", height)  # 获取高度值
    if ret != 0:
        print(f"Get height failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return

    print(f"Current resolution: Width = {width.nCurValue}, Height = {height.nCurValue}")  # 打印当前分辨率信息

    # 获取并打印当前像素格式
    stEnumValue = MVCC_ENUMVALUE()  # 定义获取枚举值的对象
    ret = cam.MV_CC_GetEnumValue("PixelFormat", stEnumValue)  # 获取像素格式值
    if ret != 0:
        print(f"Get pixel format failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return

    print(f"Current pixel format: 0x{stEnumValue.nCurValue:x}")  # 打印当前像素格式的十六进制值

    # 设置新的分辨率和像素格式（根据需要进行调整）
    new_width = 3072  # 新的宽度值
    new_height = 2048  # 新的高度值
    new_pixel_format = 0x110000d  # 新的像素格式值（替换为你想使用的像素格式）

    ret = cam.MV_CC_SetIntValue("Width", new_width)  # 设置新的宽度
    if ret != 0:
        print(f"Set width failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return

    ret = cam.MV_CC_SetIntValue("Height", new_height)  # 设置新的高度
    if ret != 0:
        print(f"Set height failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return

    ret = cam.MV_CC_SetEnumValue("PixelFormat", new_pixel_format)  # 设置新的像素格式
    if ret != 0:
        print(f"Set pixel format failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return

    print("Resolution and pixel format set successfully")  # 打印设置成功的消息

    # 开始采集
    ret = cam.MV_CC_StartGrabbing()  # 开始采集图像
    if ret != 0:
        print(f"Start grabbing failed! ret[0x{ret:x}]")  # 打印错误信息并返回
        return

    data_buf = None  # 初始化图像数据缓冲区
    buf_size = new_width * new_height * 3  # RGB8 格式的缓冲区大小
    while True:
        stFrameInfo = MV_FRAME_OUT_INFO_EX()  # 定义帧信息对象
        data_buf = (c_ubyte * buf_size)()  # 分配图像数据缓冲区
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, buf_size, stFrameInfo, 1000)  # 获取一帧图像数据
        if ret == 0:
            print(f"Get one frame: Width[{stFrameInfo.nWidth}], Height[{stFrameInfo.nHeight}], FrameNum[{stFrameInfo.nFrameNum}]")  # 打印获取的帧信息
            print(f"Frame length: {stFrameInfo.nFrameLen}")  # 打印帧数据长度
            img_buff = (c_ubyte * stFrameInfo.nFrameLen).from_address(addressof(data_buf))  # 将数据缓冲区转换为图像数据
            img = np.frombuffer(img_buff, dtype=np.uint8)  # 转换为 NumPy 数组

            print(f"Image buffer size: {img.size}")  # 打印图像数据缓冲区大小
            print(f"Expected buffer size: {new_width * new_height * 3}")  # 打印预期的缓冲区大小

            # 确定图像形状，根据像素格式调整通道数
            if new_pixel_format == 0x110000d:  # RGB8 格式
                img = img.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))  # 调整图像形状为高度、宽度、通道数
            elif new_pixel_format == 0x1080009:  # Mono8 单通道灰度图像
                img = img.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))  # 调整图像形状为高度、宽度
            elif new_pixel_format in {0x10c0027, 0x10c002b}:  # BayerRG8、BayerGB8 格式
                img = img.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))  # 调整图像形状为高度、宽度
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB if new_pixel_format == 0x10c0027 else cv2.COLOR_BAYER_BG2RGB)  # 转换为 RGB
            elif new_pixel_format == 0x1100011:  # YUV422Packed 格式
                img = img.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 2))  # 调整图像形状为高度、宽度、2个通道
                img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)  # 转换为 RGB
            else:
                print("Unsupported pixel format")  # 打印不支持的像素格式信息
                break  # 结束循环

            # 显示图像
            cv2.imshow('Camera', img)  # 在窗口中显示图像
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 等待按键事件，如果按下 'q' 键则退出循环
                break
        else:
            print(f"Get image failed! ret[0x{ret:x}]")  # 打印错误信息并结束循环
            break

    # 停止采集
    cam.MV_CC_StopGrabbing()  # 停止采集图像
    cam.MV_CC_CloseDevice()  # 关闭设备
    cam.MV_CC_DestroyHandle()  # 销毁句柄

    # 释放资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

if __name__ == "__main__":
    main()
