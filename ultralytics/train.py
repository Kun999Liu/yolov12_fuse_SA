# -*- coding: utf-8 -*-
# @Time    : 2025/8/30 12:13
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : train_yolov11.py
# @Software: PyCharm

"""
Describe:
"""
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations.check_version")

#  Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12m.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=r"F:\my_code\yolov12\ultralytics\cfg\datasets\TransmissionTower.yaml",
                      epochs=1,
                      imgsz=416,
                      device="0",
                      batch=16,
                      name="TransmissionTower",
                      amp=False,
                      pretrained=False,  # 不加载官方权重
                      )

# import torch
# print(torch.version.cuda)        # 查看编译时的 CUDA 版本
# print(torch.cuda.is_available()) # 查看是否能用 CUDA
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
# import albumentations as A
# help(A.ImageCompression)
'''测试git上传'''
