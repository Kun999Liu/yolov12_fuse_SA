import os
import numpy as np
from osgeo import gdal
from pathlib import Path
from PIL import Image


def convert_4band_to_3band_geotiff(input_folder, output_folder, bands_to_keep=[1, 2, 3], stretch_method='percentile'):
    """
    将4波段遥感影像转换为3波段PNG图像

    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        bands_to_keep: 要保留的波段索引列表（从1开始），默认保留前3个波段
                      对于GF2: [1,2,3] = Blue, Green, Red
        stretch_method: 拉伸方法
                       'percentile': 2%-98%分位数拉伸（推荐，适合遥感影像）
                       'minmax': 最小最大值拉伸
    """

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有TIFF文件
    input_path = Path(input_folder)
    tif_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))

    if not tif_files:
        print(f"在 {input_folder} 中未找到TIFF文件")
        return

    print(f"找到 {len(tif_files)} 个TIFF文件")

    # 处理每个影像
    for tif_file in tif_files:
        try:
            print(f"\n处理: {tif_file.name}")

            # 打开影像
            dataset = gdal.Open(str(tif_file))
            if dataset is None:
                print(f"  无法打开文件")
                continue

            # 获取波段数
            band_count = dataset.RasterCount
            print(f"  波段数: {band_count}")

            if band_count != 4:
                print(f"  跳过: 不是4波段影像")
                dataset = None
                continue

            # 获取影像信息
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            projection = dataset.GetProjection()
            geotransform = dataset.GetGeoTransform()

            print(f"  尺寸: {width} x {height}")

            # 读取要保留的波段
            bands_data = []
            for band_idx in bands_to_keep:
                band = dataset.GetRasterBand(band_idx)
                band_array = band.ReadAsArray()
                bands_data.append(band_array)
                print(
                    f"  读取波段 {band_idx}: 数据类型={band_array.dtype}, 范围=[{band_array.min()}, {band_array.max()}]")

            # 堆叠波段
            output_array = np.stack(bands_data, axis=0)

            # 创建输出文件（PNG格式）
            output_filename = tif_file.stem + '.png'
            output_path = Path(output_folder) / output_filename

            # 数据归一化到0-255
            # 对每个波段分别归一化
            output_8bit = np.zeros((3, height, width), dtype=np.uint8)
            for i in range(3):
                band_data = output_array[i].astype(np.float32)

                if stretch_method == 'percentile':
                    # 计算2%和98%分位数用于拉伸（避免异常值影响）
                    p2, p98 = np.percentile(band_data, (2, 98))
                    if p98 > p2:
                        band_stretched = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                    else:
                        band_stretched = band_data
                    print(f"  波段 {bands_to_keep[i]} 拉伸: [{p2:.1f}, {p98:.1f}] → [0, 255]")
                else:  # minmax
                    vmin, vmax = band_data.min(), band_data.max()
                    if vmax > vmin:
                        band_stretched = np.clip((band_data - vmin) / (vmax - vmin) * 255, 0, 255)
                    else:
                        band_stretched = band_data
                    print(f"  波段 {bands_to_keep[i]} 拉伸: [{vmin:.1f}, {vmax:.1f}] → [0, 255]")

                output_8bit[i] = band_stretched.astype(np.uint8)

            # 转换为HWC格式（Height, Width, Channels）
            output_rgb = np.transpose(output_8bit, (1, 2, 0))

            # 使用PIL保存为PNG
            from PIL import Image
            img = Image.fromarray(output_rgb, mode='RGB')
            img.save(str(output_path), 'PNG', compress_level=6)

            # 关闭数据集
            dataset = None

            print(f"  ✓ 已保存: {output_path}")

        except Exception as e:
            print(f"  ✗ 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n=" * 50)
    print("转换完成!")
    print(f"输出目录: {output_folder}")


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"C:\Users\liuku\Desktop\images"  # 输入文件夹
    output_folder = r"C:\Users\liuku\Desktop\images_rgb"  # 输出文件夹

    # GF2卫星波段说明:
    # 波段1: Blue (蓝色, 450-520nm)
    # 波段2: Green (绿色, 520-590nm)
    # 波段3: Red (红色, 630-690nm)
    # 波段4: Near Infrared (近红外, 770-890nm)

    # 转换为RGB PNG图像（保留波段1,2,3）
    # print("=" * 50)
    # print("GF2 4波段 → 3波段PNG 转换器")
    # print("=" * 50)

    convert_4band_to_3band_geotiff(
        input_folder,
        output_folder,
        bands_to_keep=[3, 2, 1],  # 保留蓝、绿、红波段
        stretch_method='percentile'  # 使用分位数拉伸获得更好的视觉效果
    )

    print("\n提示: 输出为PNG格式，使用了对比度拉伸以获得更好的显示效果")