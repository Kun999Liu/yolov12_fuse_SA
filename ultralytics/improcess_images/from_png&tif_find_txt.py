# -*- coding: utf-8 -*-
# @Time    : 2025/8/28 21:21
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : from_png&tif_find_txt.py
# @Software: PyCharm

"""
Describe:
"""
import os
import shutil

# 文件夹路径（自己修改成实际路径）
txt_folder = r"C:\Users\liuku\Desktop\GF2\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse_tiles_png_labels"
png_folder = r"C:\Users\liuku\Desktop\GF2\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse_tiles_png"
output_folder = r"C:\Users\liuku\Desktop\GF2\images"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取 txt 文件名（去掉后缀）
txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if f.endswith(".txt")}

# 遍历 png 文件夹，查找同名文件
for file in os.listdir(png_folder):
    if file.endswith(".png") or file.endswith(".tif"):
        name = os.path.splitext(file)[0]
        if name in txt_files:  # 如果有同名 txt 文件
            src = os.path.join(png_folder, file)
            dst = os.path.join(output_folder, file)
            shutil.copy2(src, dst)  # 复制文件

print("处理完成！同名文件已保存到:", output_folder)
