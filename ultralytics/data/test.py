# -*- coding: utf-8 -*-
# @Time    : 2025/9/27 15:54
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : test.py
# @Software: PyCharm

"""
Describe:
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ultralytics.data.utils import read_image

path = r"F:\my_code\datasets\train\images\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse_1_52.tif"
im_width, im_height, im_bands, projection, geotrans, im = read_image(path, 'tif')
print(type(im), im.shape, im.dtype)

import sys
import torch

# Python 版本
print("Python Version:", sys.version)

# PyTorch 版本
print("PyTorch Version:", torch.__version__)

# CUDA 可用性
print("CUDA Available:", torch.cuda.is_available())

# CUDA 设备数量
print("Number of CUDA devices:", torch.cuda.device_count())

# 每个 CUDA 设备信息
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# PyTorch 编译时 CUDA 版本
print("PyTorch CUDA Version:", torch.version.cuda)


