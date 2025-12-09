import os
import numpy as np
from osgeo import gdal
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter  # 新增 ImageEnhance 和 ImageFilter


def convert_4band_to_3band_enhanced(input_folder, output_folder, bands_to_keep=[3, 2, 1],
                                    lower_percent=0.5, upper_percent=99.5,
                                    gamma=1.2, saturation=1.3, sharpen=True):
    """
    将4波段遥感影像转换为3波段PNG图像，并进行清晰度增强。

    参数:
        input_folder: 输入文件夹
        output_folder: 输出文件夹
        bands_to_keep: 波段顺序 [Red, Green, Blue]。GF2通常是 [3, 2, 1]
        lower_percent/upper_percent: 动态范围拉伸的百分比 (默认0.5%-99.5%，比2-98保留更多细节)
        gamma: Gamma校正值 (值 < 1 变亮, > 1 变暗/增加对比度。通常 0.8-1.2 之间调整)
        saturation: 饱和度倍数 (1.0为原图，>1.0 增加色彩鲜艳度)
        sharpen: 是否应用锐化滤镜
    """

    os.makedirs(output_folder, exist_ok=True)
    input_path = Path(input_folder)
    tif_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))

    if not tif_files:
        print(f"在 {input_folder} 中未找到TIFF文件")
        return

    print(f"找到 {len(tif_files)} 个TIFF文件，开始增强处理...")

    for tif_file in tif_files:
        try:
            print(f"\n处理: {tif_file.name}")
            dataset = gdal.Open(str(tif_file))
            if dataset is None: continue

            # 读取RGB波段数据
            bands_data = []
            for band_idx in bands_to_keep:
                band = dataset.GetRasterBand(band_idx)
                if band is None:
                    raise ValueError(f"无法获取波段 {band_idx}")
                bands_data.append(band.ReadAsArray())

            height, width = bands_data[0].shape

            # --- 步骤1: 改进的百分比拉伸 ---
            output_8bit = np.zeros((3, height, width), dtype=np.uint8)

            for i, band_data in enumerate(bands_data):
                band_float = band_data.astype(np.float32)

                # 排除0值（通常是背景）不参与统计，避免拉伸错误
                valid_mask = band_float > 0
                if valid_mask.any():
                    p_low, p_high = np.percentile(band_float[valid_mask], (lower_percent, upper_percent))
                else:
                    p_low, p_high = 0, 255

                # 避免分母为0
                if p_high > p_low:
                    # 线性拉伸并截断
                    stretched = (band_float - p_low) / (p_high - p_low)
                    stretched = np.clip(stretched, 0, 1)

                    # --- 步骤2: Gamma 校正 (非线性增强) ---
                    # Gamma < 1.0 会提亮暗部，Gamma > 1.0 会压暗并增加对比度
                    # 对于卫星影像，通常先归一化，再做Gamma
                    if gamma != 1.0:
                        stretched = np.power(stretched, 1.0 / gamma)

                    output_8bit[i] = (stretched * 255).astype(np.uint8)
                else:
                    output_8bit[i] = band_float.astype(np.uint8)

            dataset = None  # 释放显存/内存

            # 转换为 PIL Image 对象
            output_rgb = np.transpose(output_8bit, (1, 2, 0))
            img = Image.fromarray(output_rgb, mode='RGB')

            # --- 步骤3: PIL 后处理增强 ---

            # A. 提升色彩饱和度 (去除雾霾感)
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation)
                print(f"  色彩增强: {saturation}x")

            # B. 提升对比度 (可选，如果Gamma不够的话)
            # enhancer = ImageEnhance.Contrast(img)
            # img = enhancer.enhance(1.1)

            # C. 锐化处理 (Unsharp Mask)
            if sharpen:
                # 使用非锐化掩模(Unsharp Mask)比普通的 SHARPEN 滤镜控制力更好
                # radius: 模糊半径, percent: 锐化强度, threshold: 阈值
                img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
                print(f"  已应用锐化滤镜")

            # 保存
            output_filename = tif_file.stem + '_enhanced.png'
            output_path = Path(output_folder) / output_filename
            img.save(str(output_path), 'PNG', compress_level=6)

            print(f"  ✓ 已保存: {output_path}")

        except Exception as e:
            print(f"  ✗ 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("增强转换完成!")


if __name__ == "__main__":
    # 路径设置
    input_folder = r"C:\Users\liuku\Desktop\datasets\images"
    output_folder = r"C:\Users\liuku\Desktop\datasets\images_rgb"

    # 调用函数
    convert_4band_to_3band_enhanced(
        input_folder,
        output_folder,
        bands_to_keep=[3, 2, 1],  # RGB顺序

        # --- 关键调整参数 ---
        lower_percent=1.0,  # 稍微放宽一点下限，去除极黑噪点
        upper_percent=99.0,  # 稍微放宽一点上限，去除极亮噪点
        gamma=1.1,  # 1.0-1.2之间尝试。如果图片太黑，尝试改为 0.8
        saturation=1.4,  # 增加饱和度，让色彩更鲜艳
        sharpen=True  # 开启锐化
    )