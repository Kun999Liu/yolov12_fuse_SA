# -*- coding: utf-8 -*-
# @Time    : 2025/12/9 21:42
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : labels_in_images.py
# @Software: PyCharm

"""
Describe:
"""
import os
import glob
import math
import numpy as np
import matplotlib
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")
# 强制使用非交互式后端，防止大量绘图时 PyCharm 崩溃
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from osgeo import gdal

# === 配置区域 ===
THUMB_SIZE = 256  # 缩略图大小
GRID_SIZE = 9  # 9x9 网格
IMAGES_PER_PAGE = GRID_SIZE * GRID_SIZE


# === 图像处理函数 ===
def linear_stretch(data):
    """ 2%-98% 线性拉伸 + 归一化 """
    data = np.nan_to_num(data)
    if np.max(data) == np.min(data):
        return data
    p2, p98 = np.percentile(data, (2, 98))
    img_clip = np.clip(data, p2, p98)
    img_norm = (img_clip - p2) / (p98 - p2)
    return np.clip(img_norm, 0, 1)


def read_gf2_rgb(tif_path):
    """ 读取 GF-2 (Band 3,2,1) """
    ds = gdal.Open(tif_path)
    if not ds: return None
    try:
        w, h = ds.RasterXSize, ds.RasterYSize
        r = ds.GetRasterBand(3).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        g = ds.GetRasterBand(2).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        b = ds.GetRasterBand(1).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        return np.dstack((linear_stretch(r), linear_stretch(g), linear_stretch(b)))
    except:
        return None


def read_gf3_gray(tif_path):
    """ 读取 GF-3 (Band 1) """
    ds = gdal.Open(tif_path)
    if not ds: return None
    try:
        w, h = ds.RasterXSize, ds.RasterYSize
        band1 = ds.GetRasterBand(1).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        return linear_stretch(band1)
    except:
        return None


# === 关键修改：解析类别 ID ===
def get_yolo_boxes(label_path):
    """
    读取 YOLO txt。
    返回: [[class_id, x_c, y_c, w, h], ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            # 确保至少有5个数据 (class + 4 coords)
            if len(parts) >= 5:
                class_id = int(parts[0])  # 第一个是类别
                coords = [float(x) for x in parts[1:5]]  # 后四个是坐标
                # 将类别和坐标合并存储
                boxes.append([class_id] + coords)
    return boxes


def plot_paginated_with_labels(image_folder, label_folder, output_dir, satellite_type="GF2", save_prefix="Checked"):
    tif_files = glob.glob(os.path.join(image_folder, "*.tif"))
    total_files = len(tif_files)

    if total_files == 0:
        print(f"[{satellite_type}] 未找到 TIF 文件: {image_folder}")
        return

    total_pages = math.ceil(total_files / IMAGES_PER_PAGE)
    print(f"--- 处理 {satellite_type} (含类别ID显示) ---")

    for page in range(total_pages):
        start_idx = page * IMAGES_PER_PAGE
        end_idx = min((page + 1) * IMAGES_PER_PAGE, total_files)
        current_batch = tif_files[start_idx:end_idx]

        fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(24, 24))
        axes = axes.flatten()

        print(f"  正在生成第 {page + 1}/{total_pages} 页...")

        for i in range(IMAGES_PER_PAGE):
            ax = axes[i]
            if i < len(current_batch):
                img_path = current_batch[i]
                file_name = os.path.basename(img_path)
                file_base = os.path.splitext(file_name)[0]
                label_path = os.path.join(label_folder, file_base + ".txt")

                # 读取图像
                img_data = None
                if satellite_type == "GF2":
                    img_data = read_gf2_rgb(img_path)
                elif satellite_type == "GF3":
                    img_data = read_gf3_gray(img_path)

                if img_data is not None:
                    if satellite_type == "GF3":
                        ax.imshow(img_data, cmap='gray', vmin=0, vmax=1)
                    else:
                        ax.imshow(img_data)

                    # === 绘制带类别的框 ===
                    if os.path.exists(label_path):
                        boxes = get_yolo_boxes(label_path)
                        for box_data in boxes:
                            # class_id = box_data[0]  # 取出类别
                            xc, yc, w, h = box_data[1:]  # 取出坐标

                            # 坐标转换 (YOLO归一化 -> 像素)
                            x_pixel = (xc - w / 2) * THUMB_SIZE
                            y_pixel = (yc - h / 2) * THUMB_SIZE
                            w_pixel = w * THUMB_SIZE
                            h_pixel = h * THUMB_SIZE

                            # 画框
                            rect = patches.Rectangle(
                                (x_pixel, y_pixel), w_pixel, h_pixel,
                                linewidth=1.5, edgecolor='red', facecolor='none'
                            )
                            ax.add_patch(rect)

                            # === 新增：在框的左上角显示类别 ID ===
                            # 使用黄色字体背景，防止看不清
                            # f"id:{class_id}",
                            ax.text(
                                x_pixel, y_pixel - 5,
                                "",
                                color='white', fontsize=6, fontweight='bold',
                                bbox=dict(facecolor='red', alpha=0.5, pad=0.5, edgecolor='none')
                            )

                        ax.set_title(
                            os.path.basename(img_path).split('_')[2] + "_" + os.path.basename(img_path).split('_')[
                                3] + "_" + os.path.basename(img_path).split('_')[7] + "_" +
                            os.path.basename(img_path).split('_')[8], fontsize=7, wrap=True, color='black')
                    else:
                        ax.set_title(
                            os.path.basename(img_path).split('_')[2] + "_" + os.path.basename(img_path).split('_')[
                                3] + "_" + os.path.basename(img_path).split('_')[7] + "_" +
                            os.path.basename(img_path).split('_')[8], fontsize=7, wrap=True, color='black')
                else:
                    ax.text(0.5, 0.5, "Error", ha='center')
            ax.axis('off')

        plt.tight_layout()
        save_path = f"{save_prefix}_{satellite_type}_Page{page + 1}.png"
        plt.savefig(output_dir + "/" + save_path, dpi=150)
        plt.close(fig)
        print(f"  ✅ 已保存: {save_path}")


if __name__ == "__main__":
    # === 请修改这里的路径 ===
    base_dir = r"C:\Users\liuku\Desktop"
    output_dir = r"./checked_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gf2_img = os.path.join(base_dir, "images")
    gf2_lbl = os.path.join(base_dir, "labels")

    # gf3_img = os.path.join(base_dir, "GF3_Images")
    # gf3_lbl = os.path.join(base_dir, "GF3_Labels")

    if os.path.exists(gf2_img):
        plot_paginated_with_labels(gf2_img, gf2_lbl, output_dir, "GF2")

    # if os.path.exists(gf3_img):
    #     plot_paginated_with_labels(gf3_img, gf3_lbl, "GF3")
