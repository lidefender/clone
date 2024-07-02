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
    width = MVCC_INTVALUE()
    height = MVCC_INTVALUE()
    ret = cam.MV_CC_GetIntValue("Width", width)
    if ret != 0:
        print(f"Get width failed! ret[0x{ret:x}]")
        return
    ret = cam.MV_CC_GetIntValue("Height", height)
    if ret != 0:
        print(f"Get height failed! ret[0x{ret:x}]")
        return

    print(f"Current resolution: Width = {width.nCurValue}, Height = {height.nCurValue}")

    # 获取并打印当前像素格式
    stEnumValue = MVCC_ENUMVALUE()
    ret = cam.MV_CC_GetEnumValue("PixelFormat", stEnumValue)
    if ret != 0:
        print(f"Get pixel format failed! ret[0x{ret:x}]")
        return

    print(f"Current pixel format: 0x{stEnumValue.nCurValue:x}")

    # 设置新的分辨率和像素格式（根据需要进行调整）
    new_width = 1920
    new_height = 1080
    new_pixel_format = 0x110000d  # 替换为你想使用的像素格式

    ret = cam.MV_CC_SetIntValue("Width", new_width)
    if ret != 0:
        print(f"Set width failed! ret[0x{ret:x}]")
        return

    ret = cam.MV_CC_SetIntValue("Height", new_height)
    if ret != 0:
        print(f"Set height failed! ret[0x{ret:x}]")
        return

    ret = cam.MV_CC_SetEnumValue("PixelFormat", new_pixel_format)
    if ret != 0:
        print(f"Set pixel format failed! ret[0x{ret:x}]")
        return

    print("Resolution and pixel format set successfully")

    # 开始采集
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"Start grabbing failed! ret[0x{ret:x}]")
        return

    data_buf = None
    buf_size = new_width * new_height * 3  # 根据新的分辨率和像素格式设置缓冲区大小
    while True:
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        data_buf = (c_ubyte * buf_size)()
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, buf_size, stFrameInfo, 1000)
        if ret == 0:
            print(f"Get one frame: Width[{stFrameInfo.nWidth}], Height[{stFrameInfo.nHeight}], FrameNum[{stFrameInfo.nFrameNum}]")
            img_buff = (c_ubyte * stFrameInfo.nFrameLen).from_address(addressof(data_buf))
            img = np.frombuffer(img_buff, dtype=np.uint8)
            img = img.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))

            # 显示图像
            cv2.imshow('Camera', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Get image failed! ret[0x{ret:x}]")
            break

    # 停止采集
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()

    # 释放资源
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
