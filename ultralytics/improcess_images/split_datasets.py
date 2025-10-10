# -*- coding: utf-8 -*-
# @Time    : 2025/8/30 19:03
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : split_datasets.py
# @Software: PyCharm

"""
Describe:
"""
import os
import shutil
import random

# 设置路径
images_dir = r"C:\Users\liuku\Desktop\GF2\images"
labels_dir = r"C:\Users\liuku\Desktop\GF2\labels"

output_dir = r"C:\Users\liuku\Desktop\GF2\datasets"  # 输出总文件夹

# 设置划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 创建输出文件夹结构
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

# 获取所有图片文件名并打乱顺序
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
random.shuffle(image_files)

# 划分数量
num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)
num_test = num_images - num_train - num_val  # 防止四舍五入问题

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]


# 复制文件函数
def copy_files(file_list, split):
    for file_name in file_list:
        # 复制图片
        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(output_dir, split, "images", file_name)
        shutil.copy2(src_img, dst_img)

        # 对应的标签
        label_name = os.path.splitext(file_name)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(output_dir, split, "labels", label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"Warning: Label not found for {file_name}")


# 执行划分
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("数据集三划分完成！")
