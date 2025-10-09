# -*- coding: utf-8 -*-
# @Time    : 2025/8/30 12:13
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : train_yolov12.py
# @Software: PyCharm

"""
Describe:
"""
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations.check_version")
warnings.simplefilter(action='ignore', category=FutureWarning)

#  Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12m.pt")
# data=r"F:\my_code\yolov12\ultralytics\cfg\datasets\TransmissionTower.yaml",
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=r'D:\Git\yolov12_fuse_SA\ultralytics\cfg\datasets\TransmissionTower.yaml',
                      epochs=500,
                      patience=150,
                      imgsz=416,
                      workers=8,
                      device="1",
                      batch=16,
                      name="TransmissionTower_3bands_M_500epochs",
                      amp=False,
                      pretrained=False,  # 不加载官方权重
                      )


# from ultralytics.data.utils import read_image
# path = r"D:\yolodatasets\datasets_2m\train\images\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse_8_28.tif"
# im_width, im_height, im_bands, projection, geotrans, im = read_image(path, 'tif', bands=4)
# print(im_width, im_height, im_bands, projection, geotrans, im)
# print(type(im), im.shape, im.dtype)
# import sys
# import torch
#
# # Python 版本
# print("Python Version:", sys.version)
#
# # PyTorch 版本
# print("PyTorch Version:", torch.__version__)
#
# # CUDA 可用性
# print("CUDA Available:", torch.cuda.is_available())
#
# # CUDA 设备数量
# print("Number of CUDA devices:", torch.cuda.device_count())
#
# # 每个 CUDA 设备信息
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
#
# # PyTorch 编译时 CUDA 版本
# print("PyTorch CUDA Version:", torch.version.cuda)
# '''测试git上传'''

'''测试git上传--Aircas_laptop'''
