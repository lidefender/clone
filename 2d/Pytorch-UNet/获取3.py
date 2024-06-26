import ctypes
import cv2
import os
import msvcrt
import numpy as np

# 加载MVS SDK库
dll_path = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll"
mv_dll = ctypes.CDLL(dll_path)

# 定义相关常量和结构体
MV_OK = 0
MV_CC_PIXEL_TYPE_GVSP_BGR8_PACKED = 0x02180015


class MV_CC_DEVICE_INFO(ctypes.Structure):
    _fields_ = [("nTLayerType", ctypes.c_uint),
                ("SpecialInfo", ctypes.c_ubyte * 512)]


class MV_FRAME_OUT_INFO_EX(ctypes.Structure):
    _fields_ = [("nWidth", ctypes.c_uint),
                ("nHeight", ctypes.c_uint),
                ("enPixelType", ctypes.c_uint),
                ("nFrameNum", ctypes.c_uint),
                ("nDevTimeStampHigh", ctypes.c_uint),
                ("nDevTimeStampLow", ctypes.c_uint),
                ("nHostTimeStamp", ctypes.c_ulonglong),
                ("nFrameLen", ctypes.c_uint),
                ("Reserved", ctypes.c_uint * 4)]


class MV_CC_PIXEL_CONVERT_PARAM(ctypes.Structure):
    _fields_ = [("nWidth", ctypes.c_uint),
                ("nHeight", ctypes.c_uint),
                ("enSrcPixelType", ctypes.c_uint),
                ("pSrcData", ctypes.POINTER(ctypes.c_ubyte)),
                ("nSrcDataLen", ctypes.c_uint),
                ("enDstPixelType", ctypes.c_uint),
                ("pDstBuffer", ctypes.POINTER(ctypes.c_ubyte)),
                ("nDstBufferSize", ctypes.c_uint),
                ("nDstDataLen", ctypes.POINTER(ctypes.c_uint)),
                ("nReserved", ctypes.c_uint)]


def setup_camera():
    # 初始化相机
    handle = ctypes.c_void_p()
    ret = mv_dll.MV_CC_CreateHandle(ctypes.byref(handle), ctypes.byref(MV_CC_DEVICE_INFO()))
    if ret != MV_OK:
        raise Exception("无法创建句柄")

    # 打开设备
    ret = mv_dll.MV_CC_OpenDevice(handle)
    if ret != MV_OK:
        raise Exception("无法打开设备")

    # 设置像素格式
    ret = mv_dll.MV_CC_SetEnumValue(handle, "PixelFormat", MV_CC_PIXEL_TYPE_GVSP_BGR8_PACKED)
    if ret != MV_OK:
        raise Exception("无法设置像素格式")

    # 开始采集
    ret = mv_dll.MV_CC_StartGrabbing(handle)
    if ret != MV_OK:
        raise Exception("无法开始采集")

    return handle


def capture_and_save_images(handle, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    print("按 's' 键保存当前帧，按 'q' 键退出程序")

    frame_info = MV_FRAME_OUT_INFO_EX()
    data_buf = (ctypes.c_ubyte * (1920 * 1080 * 3))()  # 假设图像分辨率为1920x1080
    while True:
        ret = mv_dll.MV_CC_GetOneFrameTimeout(handle, data_buf, len(data_buf), ctypes.byref(frame_info), 1000)
        if ret != MV_OK:
            print("无法读取帧")
            continue

        # 将缓冲区数据转换为OpenCV格式
        frame = np.ctypeslib.as_array(data_buf).reshape((frame_info.nHeight, frame_info.nWidth, 3))

        # 显示当前帧
        cv2.imshow('Live Video', frame)

        # 检查键盘输入
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()

            if key == 's':
                frame_count += 1
                filename = os.path.join(output_dir, f"frame_{frame_count}.png")
                cv2.imwrite(filename, frame)
                print(f"帧已保存: {filename}")
            elif key == 'q':
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mv_dll.MV_CC_StopGrabbing(handle)
    mv_dll.MV_CC_CloseDevice(handle)
    mv_dll.MV_CC_DestroyHandle(handle)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    output_dir = "captured_frames"
    handle = setup_camera()
    capture_and_save_images(handle, output_dir)
