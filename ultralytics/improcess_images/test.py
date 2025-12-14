# import ultralytics
# print(ultralytics.__version__)
# print(ultralytics.__file__)
#
# from ultralytics.data.utils import read_image
# img = read_image(r"F:\wxy_code\mydata_7bands\images\train\GF2_PMS1_E82.7_N45.2_20220430_L1A0006441524-pansharpen2colNum_12rowNum_39.npy", "npy")
# print(img.shape[0], img.shape[1], img.shape[2])

import os
import glob
import numpy as np
import rasterio
import xml.etree.ElementTree as ET

# ================= 用户配置区 =================
# 1. 这里放你那个“数值偏低”的 dB 影像文件夹
INPUT_DB_FOLDER = r"E:\GF3\image"

# 2. 这里放原始数据的 XML 文件夹 (用于读取参数)
# 脚本会根据文件名尝试自动匹配
XML_FOLDER = r"E:\GF3\GF3-tar\xml"

# 3. 结果保存位置
OUTPUT_FOLDER = r"E:\GF3\image"


# ===========================================

def get_db_offset(xml_path, polarization='HH'):
    """
    根据 QualifyValue 计算需要补回的 dB 增益量
    公式: Offset = 20 * log10( 32767 / QualifyValue )
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 1. 读取 QualifyValue
        # 注意：不同极化(HH/HV)的Qualify值可能不同，这里默认读HH
        q_path = f".//QualifyValue/{polarization}"
        q_node = root.find(q_path)

        if q_node is None or q_node.text == 'NULL':
            print(f"  [警告] XML中未找到 {polarization} 的 QualifyValue，默认使用偏移量 0")
            return 0.0

        q_value = float(q_node.text)

        # 2. 确定最大量化值 (通常是 32767)
        bit_node = root.find(".//imageinfo/imagebit")
        max_val = 32767.0
        if bit_node is not None and "32" in bit_node.text:
            max_val = 2147483647.0  # 极少数情况

        # 3. 计算正向补偿量
        # 比如 Q=598, M=32767 -> 32767/598 ≈ 54.7倍
        # 转dB: 20 * log10(54.7) ≈ 34.76 dB
        offset = 20 * np.log10(max_val / q_value)

        return offset

    except Exception as e:
        print(f"  [错误] 读取XML失败: {e}")
        return 0.0


def apply_correction(tif_path, xml_path, out_path):
    print(f"正在处理: {os.path.basename(tif_path)}")

    # 自动计算修正值
    offset = get_db_offset(xml_path, polarization='HH')

    if offset == 0:
        print("  -> 修正量为 0，跳过处理 (或检查XML)")
        return

    print(f"  -> 计算出的补偿量: +{offset:.4f} dB")

    # 执行加法运算
    with rasterio.open(tif_path) as src:
        # 读取数据 (假设是单波段 dB 数据)
        db_data = src.read(1)
        profile = src.profile.copy()

        # 处理无效值 (NoData)
        # 如果你的背景是 0 或 NaN，要保持它们不变
        nodata_val = src.nodata
        if nodata_val is None:
            nodata_val = np.nan  # 默认假设 NaN

        # 创建输出数组
        corrected_data = db_data.copy()

        # 仅对有效像素进行加法修正
        # 逻辑：如果不是NaN 且 (如果定义了nodata则不等于nodata)
        if np.isnan(nodata_val):
            valid_mask = ~np.isnan(db_data)
        else:
            valid_mask = (db_data != nodata_val)

        # 核心操作：直接相加
        # -30 + 34.76 = +4.76
        corrected_data[valid_mask] = db_data[valid_mask] + offset

        # 保存
        profile.update(dtype=rasterio.float32, compress='lzw')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(corrected_data, 1)

    print(f"  -> 已保存: {os.path.basename(out_path)}\n")


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 获取所有 TIF 文件
    tif_files = glob.glob(os.path.join(INPUT_DB_FOLDER, "*.tif"))

    if not tif_files:
        print("未找到TIF文件，请检查 INPUT_DB_FOLDER 路径")
        return

    for tif_file in tif_files:
        # 尝试寻找同名的 XML (忽略后缀差异)
        # 比如 image.tif -> image.meta.xml 或 image.xml
        basename = os.path.splitext(os.path.basename(tif_file))[0]

        # 这里需要你根据实际文件名规律调整
        # 假设原始XML名包含在tif文件名中，或者完全一致
        # 简单匹配策略：在 XML 文件夹里找 "包含 tif文件名核心部分" 的 xml
        potential_xmls = glob.glob(os.path.join(XML_FOLDER, f"*{basename}*.xml"))

        xml_path = None
        if potential_xmls:
            xml_path = potential_xmls[0]  # 取第一个匹配的

        if xml_path and os.path.exists(xml_path):
            out_name = f"{basename}_Corrected.tif"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            apply_correction(tif_file, xml_path, out_path)
        else:
            print(f"[跳过] 找不到 {basename} 对应的 XML 文件，无法计算修正量。")


if __name__ == "__main__":
    main()