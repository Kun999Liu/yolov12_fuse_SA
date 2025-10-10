# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 16:48
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : GF2_find_GF3.py
# @Software: PyCharm

"""
Describe:
"""
import shutil

from osgeo import gdal, osr
import os
import numpy as np
from tqdm import tqdm


def crop_by_small_tif(big_tif, small_folder, out_folder, nodata_threshold=0.75):
    # 打开大影像
    big_ds = gdal.Open(big_tif)
    if big_ds is None:
        raise FileNotFoundError(f"无法打开大影像: {big_tif}")

    # 获取大影像的投影和坐标系信息
    big_proj = big_ds.GetProjection()
    big_srs = osr.SpatialReference(wkt=big_proj)
    big_gt = big_ds.GetGeoTransform()

    # 计算大影像范围 (投影坐标)
    big_x_min = big_gt[0]
    big_y_max = big_gt[3]
    big_x_res = big_gt[1]
    big_y_res = big_gt[5]
    big_x_size = big_ds.RasterXSize
    big_y_size = big_ds.RasterYSize
    big_x_max = big_x_min + big_x_res * big_x_size
    big_y_min = big_y_max + big_y_res * big_y_size

    # 转换成经纬度范围
    coordTrans_big = osr.CoordinateTransformation(big_srs, big_srs.CloneGeogCS())
    big_lon_min, big_lat_min, _ = coordTrans_big.TransformPoint(big_x_min, big_y_min)
    big_lon_max, big_lat_max, _ = coordTrans_big.TransformPoint(big_x_max, big_y_max)

    print("大影像经纬度范围：")
    print(f"  左下角: ({big_lon_min:.6f}, {big_lat_min:.6f})")
    print(f"  右上角: ({big_lon_max:.6f}, {big_lat_max:.6f})")

    # 获取所有小影像文件列表
    small_files = [f for f in os.listdir(small_folder) if f.lower().endswith(".tif")]

    # 遍历小影像文件夹，加上进度条
    for fname in tqdm(small_files, desc="正在裁剪", unit="张"):
        small_path = os.path.join(small_folder, fname)
        out_path = os.path.join(out_folder, fname)

        # 打开小影像
        small_ds = gdal.Open(small_path)
        if small_ds is None:
            print(f"无法打开 {small_path}")
            continue

        # 获取小影像范围 (投影坐标)
        gt = small_ds.GetGeoTransform()
        x_min = gt[0]
        y_max = gt[3]
        x_res = gt[1]
        y_res = gt[5]
        x_size = small_ds.RasterXSize
        y_size = small_ds.RasterYSize
        x_max = x_min + x_res * x_size
        y_min = y_max + y_res * y_size

        # 转换成经纬度
        small_srs = osr.SpatialReference(wkt=small_ds.GetProjection())
        coordTrans = osr.CoordinateTransformation(small_srs, small_srs.CloneGeogCS())
        lon_min, lat_min, _ = coordTrans.TransformPoint(x_min, y_min)
        lon_max, lat_max, _ = coordTrans.TransformPoint(x_max, y_max)

        # ===== 检查是否在大影像经纬度范围内 =====
        if (lon_min < min(big_lon_min, big_lon_max) or
            lon_max > max(big_lon_min, big_lon_max) or
            lat_min < min(big_lat_min, big_lat_max) or
            lat_max > max(big_lat_min, big_lat_max)):
            print(f"⚠️ {fname} 超出大影像范围，跳过。")
            continue

        # === 如果范围合法，转换到大图坐标系裁剪 ===
        if not big_srs.IsSameGeogCS(small_srs):
            geoTrans = osr.CoordinateTransformation(small_srs.CloneGeogCS(), big_srs)
            x_min, y_min, _ = geoTrans.TransformPoint(lon_min, lat_min)
            x_max, y_max, _ = geoTrans.TransformPoint(lon_max, lat_max)

        # 执行裁剪
        gdal.Warp(
            out_path, big_ds,
            format="GTiff",
            outputBounds=(x_min, y_min, x_max, y_max),
            dstNodata=0,
            xRes=x_res,
            yRes=abs(y_res)
        )

        # ===== 检查 NoData 比例 =====
        out_ds = gdal.Open(out_path)
        if out_ds is None:
            continue

        band = out_ds.GetRasterBand(1)
        nodata_value = band.GetNoDataValue()
        data = band.ReadAsArray()

        if nodata_value is not None:
            nodata_mask = (data == nodata_value)
            nodata_ratio = np.sum(nodata_mask) / data.size
            if nodata_ratio > nodata_threshold:
                out_ds = None
                os.remove(out_path)
                print(f"⚠️ {fname} 裁剪结果超过 {nodata_threshold*100:.1f}% NoData，已删除。")
                continue

        print(f"✅ 裁剪完成: {out_path}")

    print("🎉 所有裁剪任务完成！")

def filter_by_nodata(folder, nodata_threshold=0.75):
    """
    检查文件夹下所有tif影像的NoData比例，超过阈值的文件将被删除
    :param folder: 输入文件夹路径
    :param nodata_threshold: 阈值 (默认 0.75，即 75%)
    """
    files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]

    for fname in tqdm(files, desc="检查NoData比例", unit="张"):
        fpath = os.path.join(folder, fname)

        ds = gdal.Open(fpath)
        if ds is None:
            print(f"无法打开 {fpath}")
            continue

        band = ds.GetRasterBand(1)
        nodata_value = band.GetNoDataValue()
        data = band.ReadAsArray()

        if nodata_value is None:
            # 如果没有显式NoData，就跳过检查
            print(f"{fname} 没有NoData定义，跳过。")
            continue

        nodata_mask = (data == nodata_value)
        nodata_ratio = np.sum(nodata_mask) / data.size

        if nodata_ratio > nodata_threshold:
            ds = None
            os.remove(fpath)
            print(f"⚠️ {fname} 的NoData比例 {nodata_ratio*100:.1f}% > {nodata_threshold*100:.0f}%，已删除。")
        else:
            print(f"✅ {fname} 保留 (NoData比例 {nodata_ratio*100:.1f}%)")

def copy_common_images(folder1, folder2, output_folder):
    """
    找出两个文件夹中同名(忽略后缀)的文件，
    并仅复制 folder1 中的文件到新的文件夹。
    """
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件名（不含后缀）
    files1 = {os.path.splitext(f)[0]: f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files2 = {os.path.splitext(f)[0]: f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}

    # 找出相同的文件名（忽略后缀）
    common_names = set(files1.keys()).intersection(files2.keys())

    if not common_names:
        print("未找到相同的文件（忽略后缀）。")
        return

    # 遍历并复制 folder1 中的同名文件
    for name in tqdm(common_names, desc="复制进度", unit="file"):
        src_path = os.path.join(folder1, files1[name])
        dst_path = os.path.join(output_folder, files1[name])
        shutil.copy(src_path, dst_path)

    print(f"已复制 {len(common_names)} 个相同文件到 {output_folder}")

def remove_empty_files(folder):
    """
    删除指定文件夹下大小为 0 字节的文件
    :param folder: 文件夹路径
    """
    removed_files = []

    for root, _, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
                    os.remove(fpath)
                    removed_files.append(fpath)
            except Exception as e:
                print(f"删除失败: {fpath}, 错误: {e}")

    print(f"已删除 {len(removed_files)} 个空文件。")
    if removed_files:
        print("删除的文件：")
        for f in removed_files:
            print("  " + f)

if __name__ == "__main__":
    # big_tif = r"E:\GF3_TIF\GF3_KAS_UFS_011121_E93.4_N42.6_20180920_L1A_DH_L10003466944_db_warp_warp.tif" # 大影像路径
    # small_folder = r"D:\数据\datasets\images_tif" # 小切片所在文件夹
    # out_folder = r"D:\数据\datasets\image_sar"  # 输出文件夹
    # os.makedirs(out_folder, exist_ok=True)
    # crop_by_small_tif(big_tif, small_folder, out_folder)
    # copy_common_images(r"D:\数据\datasets\labels", r"D:\数据\datasets\images_sar", r"D:\数据\datasets\images_common_labels")
    remove_empty_files(r"D:\OneDrive_files\OneDrive\GF2\labels")

