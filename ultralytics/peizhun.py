# -*- coding: utf-8 -*-
# @Time    : 2025/9/1 19:41
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : peizhun.py
# @Software: PyCharm

"""
Describe:
"""
import cv2
import numpy as np
from osgeo import gdal
from skimage.metrics import normalized_mutual_information


def read_gdal(path, band=1):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(band).ReadAsArray()
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    return arr, geo, proj


# 读取 GF2 & GF3
gf2, geo2, proj2 = read_gdal(r"F:\BaiduNetdiskDownload\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse.tif")
gf3, geo3, proj3 = read_gdal(r"F:\BaiduNetdiskDownload\GF3\tif\GF3_KAS_UFS_011121_E93.4_N42.6_20180920_L1A_DH_L10003466944_db.tif")


# 归一化
def normalize(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)


gf2_norm = normalize(gf2)
gf3_norm = normalize(gf3)

# === Step1: 网格采样候选点 ===
step = 512  # 每隔512像素取一个点
template_size = 64  # 模板窗口
search_radius = 64  # 搜索窗口半径

src_pts, dst_pts = [], []

for y in range(template_size, gf3_norm.shape[0] - template_size, step):
    for x in range(template_size, gf3_norm.shape[1] - template_size, step):
        template = gf3_norm[y - template_size:y + template_size, x - template_size:x + template_size]

        # 限定搜索区域
        y1, y2 = max(0, y - search_radius), min(gf2_norm.shape[0], y + search_radius)
        x1, x2 = max(0, x - search_radius), min(gf2_norm.shape[1], x + search_radius)
        search_area = gf2_norm[y1:y2, x1:x2]

        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            continue

        best_score = -1
        best_loc = None

        # === Step2: MI 匹配 ===
        for dy in range(0, search_area.shape[0] - template.shape[0], 4):  # 步长4加速
            for dx in range(0, search_area.shape[1] - template.shape[1], 4):
                patch = search_area[dy:dy + template.shape[0], dx:dx + template.shape[1]]
                score = normalized_mutual_information(template, patch)
                if score > best_score:
                    best_score = score
                    best_loc = (dx, dy)

        if best_score > 0.3:  # MI 阈值（0.3-0.5 之间可调）
            src_pts.append([x, y])
            dst_pts.append([x1 + best_loc[0], y1 + best_loc[1]])

src_pts = np.array(src_pts, dtype=np.float32)
dst_pts = np.array(dst_pts, dtype=np.float32)

print(f"找到控制点: {len(src_pts)}")

# === Step3: 拟合几何变换 (仿射) ===
if len(src_pts) >= 3:
    H, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
    h, w = gf3_norm.shape
    gf2_warp = cv2.warpAffine(gf2, H, (w, h))

    # === Step4: 保存结果 ===
    driver = gdal.GetDriverByName("GTiff")
    out_path = "GF2_registered_MI.tif"
    dst_ds = driver.Create(out_path, w, h, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geo3)
    dst_ds.SetProjection(proj3)
    dst_ds.GetRasterBand(1).WriteArray(gf2_warp)
    dst_ds.FlushCache()
    dst_ds = None

    print(f"✅ 配准完成，输出: {out_path}")
else:
    print("❌ 控制点不足，配准失败")


