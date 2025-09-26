from osgeo import gdal, osr
import numpy as np
from PIL import ImageEnhance, Image
import os
from tqdm import tqdm
import time

def split_image(input_image_path, output_folder, tile_size=416):
    # 打开输入影像
    input_dataset = gdal.Open(input_image_path)
    if input_dataset is None:
        print("无法打开输入影像.")
        return

    # 获取影像尺寸
    width = input_dataset.RasterXSize
    height = input_dataset.RasterYSize

    # 计算每个维度上的切片数量
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # 获取无效值
    no_data_value = input_dataset.GetRasterBand(1).GetNoDataValue()

    # 获取原始文件名（不包含扩展名）
    file_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建并打开文本文件记录经纬度
    # 获取 output_folder 的上级目录
    parent_folder = os.path.dirname(output_folder)
    lat_lon_file_path = os.path.join(parent_folder, f"{file_name}.txt")
    with open(lat_lon_file_path, 'w') as lat_lon_file:
        lat_lon_file.write("Tile Filename: Latitude-Longitude Bounds\n")

        # 循环处理每个切片
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # 计算切片坐标
                tile_x = x * tile_size
                tile_y = y * tile_size

                # 读取切片数据
                tile_data = input_dataset.ReadAsArray(tile_x, tile_y, tile_size, tile_size)

                # 处理无效值
                if no_data_value is not None:
                    tile_data[tile_data == no_data_value] = 0  # 替换无效值为0

                # 将切片写入新的GeoTIFF文件
                output_tile_path = os.path.join(output_folder, f"{file_name}_{x}_{y}.tif")
                write_tile(tile_data, input_dataset, output_tile_path, tile_x, tile_y, tile_size, lat_lon_file)

    # 关闭输入数据集
    input_dataset = None


def write_tile(tile_data, input_dataset, output_tile_path, tile_x, tile_y, tile_size, lat_lon_file):
    # 创建新的GeoTIFF文件
    driver = gdal.GetDriverByName('GTiff')
    num_bands = input_dataset.RasterCount
    output_dataset = driver.Create(output_tile_path, tile_size, tile_size, num_bands, gdal.GDT_Float32)

    # 设置投影和地理转换
    output_dataset.SetProjection(input_dataset.GetProjection())
    geotransform = list(input_dataset.GetGeoTransform())
    geotransform[0] = geotransform[0] + tile_x * geotransform[1]
    geotransform[3] = geotransform[3] + tile_y * geotransform[5]
    output_dataset.SetGeoTransform(geotransform)

    # 写入切片数据
    for band_index in range(num_bands):
        band_data = tile_data[band_index, :, :]
        output_band = output_dataset.GetRasterBand(band_index + 1)
        output_band.WriteArray(band_data)

        # 设置无效值
        no_data_value = input_dataset.GetRasterBand(band_index + 1).GetNoDataValue()
        if no_data_value is not None:
            output_band.SetNoDataValue(no_data_value)

    # 计算并记录每个切片的经纬度范围
    lat_lon_range = get_tile_lat_lon_bounds(output_dataset.GetGeoTransform(), output_dataset.GetProjection(), tile_size)
    lat_lon_info = f"{os.path.basename(output_tile_path)}: {lat_lon_range}\n"
    print(lat_lon_info)
    lat_lon_file.write(lat_lon_info)

    # 关闭输出数据集
    output_dataset = None


def get_tile_lat_lon_bounds(geotransform, projection, tile_size):
    # 设置坐标转换，从投影坐标到经纬度坐标
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection)
    lat_lon_srs = spatial_ref.CloneGeogCS()
    transform = osr.CoordinateTransformation(spatial_ref, lat_lon_srs)

    # 左上角坐标
    ulx, uly, _ = transform.TransformPoint(geotransform[0], geotransform[3])
    # 右下角坐标
    lrx, lry, _ = transform.TransformPoint(geotransform[0] + geotransform[1] * tile_size,
                                           geotransform[3] + geotransform[5] * tile_size)

    return (ulx, uly, lrx, lry)


#将  4 波段tif 转换为 3 波段png
def convert_to_rgb(input_tif, output_image, contrast_factor=2.0):
    """单张tif转换为RGB png"""
    dataset = gdal.Open(input_tif)
    if dataset is None:
        raise RuntimeError("无法打开文件")

    # 读取波段数据（假设1=蓝，2=绿，3=红）
    red_band = dataset.GetRasterBand(3).ReadAsArray().astype(np.float32)
    green_band = dataset.GetRasterBand(2).ReadAsArray().astype(np.float32)
    blue_band = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # 对波段进行缩放，以避免超出255范围
    red_band = (red_band / np.max(red_band)) * 255
    green_band = (green_band / np.max(green_band)) * 255
    blue_band = (blue_band / np.max(blue_band)) * 255

    # 创建RGB图像
    rgb = np.dstack((red_band, green_band, blue_band))

    # 转为PIL图像
    image = Image.fromarray(rgb.astype(np.uint8))

    # 图像增强
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # 保存为PNG格式
    image.save(output_image)


def batch_convert_tif_to_rgb(input_folder, output_folder, contrast_factor=2.0, resume=True):
    """批量转换tif为RGB png，支持进度条、耗时统计、失败日志、断点续跑和总结报告"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_file = os.path.join(output_folder, "processed_files.txt")
    processed_files = set()

    # 如果启用断点续跑，则读取历史日志
    if resume and os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f.readlines())

    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    total = len(tif_files)
    failed_files = []
    success_count = 0

    start_time = time.time()

    with tqdm(total=total, desc="转换进度", unit="file") as pbar:
        for idx, tif_file in enumerate(tif_files, start=1):
            input_tif = os.path.join(input_folder, tif_file)
            output_image = os.path.join(output_folder, os.path.splitext(tif_file)[0] + ".png")

            # 跳过已处理过的文件（仅在resume=True时生效）
            if resume and tif_file in processed_files:
                pbar.set_postfix_str(f"已完成 {idx}/{total} (跳过)")
                pbar.update(1)
                continue

            try:
                convert_to_rgb(input_tif, output_image, contrast_factor)
                success_count += 1
                # 仅在resume=True时记录日志
                if resume:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(tif_file + "\n")
            except Exception as e:
                failed_files.append((tif_file, str(e)))

            pbar.set_postfix_str(f"已完成 {idx}/{total}")
            pbar.update(1)

    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        print(f"\n全部转换完成！耗时 {elapsed_time:.2f} 秒")
    else:
        print(f"\n全部转换完成！耗时 {elapsed_time/60:.2f} 分钟")

    # 输出总结
    failed_count = len(failed_files)
    summary = (
        f"\n转换总结：\n"
        f"  - 总文件数: {total}\n"
        f"  - 成功数: {success_count}\n"
        f"  - 失败数: {failed_count}\n"
        f"  - 成功率: {success_count/total*100:.2f}%\n"
    )
    print(summary)

    # 写入 summary.txt
    with open(os.path.join(output_folder, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    # 失败文件单独记录
    if failed_files:
        print("以下文件转换失败（已写入 failed_files.txt）：")
        for f1, err in failed_files:
            print(f"  - {f1} (错误: {err})")
        with open(os.path.join(output_folder, "failed_files.txt"), "w", encoding="utf-8") as f:
            for fname, err in failed_files:
                f.write(f"{fname}\t{err}\n")
    else:
        print("所有文件均成功转换 ✅")


# 使用示例
if __name__ == "__main__":
    input_folder = r"C:\Users\liuku\Desktop\images"
    output_folder = r"C:\Users\liuku\Desktop\imaegs-3bands"

    # 默认断点续跑
    batch_convert_tif_to_rgb(input_folder, output_folder, contrast_factor=2.0, resume=False)

# 示例用法:
if __name__ == "__main__":
    input_image_path = r"F:\BaiduNetdiskDownload\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse.tif"
    output_folder = r"C:\Users\liuku\Desktop\images"
    split_image(input_image_path, output_folder)



# from osgeo import gdal
# import os
#
#
# def split_image(input_image_path, output_folder, tile_size=416):
#     # 打开输入影像
#     input_dataset = gdal.Open(input_image_path)
#     if input_dataset is None:
#         print("无法打开输入影像.")
#         return
#
#     # 获取影像尺寸
#     width = input_dataset.RasterXSize
#     height = input_dataset.RasterYSize
#     num_bands = input_dataset.RasterCount
#
#     # 计算每个维度上的切片数量
#     num_tiles_x = width // tile_size
#     num_tiles_y = height // tile_size
#
#     # 获取无效值
#     no_data_value = input_dataset.GetRasterBand(1).GetNoDataValue()
#
#     # 获取原始文件名（不包含扩展名）
#     file_name = os.path.splitext(os.path.basename(input_image_path))[0]
#
#     # 如果输出文件夹不存在，则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 循环处理每个切片
#     for y in range(num_tiles_y):
#         for x in range(num_tiles_x):
#             # 计算切片坐标
#             tile_x = x * tile_size
#             tile_y = y * tile_size
#
#             # 读取切片数据
#             tile_data = input_dataset.ReadAsArray(tile_x, tile_y, tile_size, tile_size)
#
#             # 处理无效值
#             if no_data_value is not None:
#                 tile_data[tile_data == no_data_value] = 0  # 替换无效值为0
#
#             # 将切片写入新的GeoTIFF文件
#             output_tile_path = os.path.join(output_folder, f"{file_name}_{x}_{y}.tif")
#             write_tile(tile_data, input_dataset, output_tile_path, tile_x, tile_y, tile_size)
#
#     # 关闭输入数据集
#     input_dataset = None
#
#
# def write_tile(tile_data, input_dataset, output_tile_path, tile_x, tile_y, tile_size):
#     # 创建新的GeoTIFF文件
#     driver = gdal.GetDriverByName('GTiff')
#     num_bands = input_dataset.RasterCount
#     output_dataset = driver.Create(output_tile_path, tile_size, tile_size, num_bands, gdal.GDT_Float32)
#
#     # 设置投影和地理转换
#     output_dataset.SetProjection(input_dataset.GetProjection())
#     geotransform = list(input_dataset.GetGeoTransform())
#     geotransform[0] = geotransform[0] + tile_x * geotransform[1]
#     geotransform[3] = geotransform[3] + tile_y * geotransform[5]
#     output_dataset.SetGeoTransform(geotransform)
#
#     # 写入切片数据
#     for band_index in range(num_bands):
#         band_data = tile_data[band_index, :, :]
#         output_band = output_dataset.GetRasterBand(band_index + 1)
#         output_band.WriteArray(band_data)
#
#         # 设置无效值
#         no_data_value = input_dataset.GetRasterBand(band_index + 1).GetNoDataValue()
#         if no_data_value is not None:
#             output_band.SetNoDataValue(no_data_value)
#
#     # 关闭输出数据集
#     output_dataset = None
#
#
# # 示例用法:
# input_image_path = (
#     r"F:\Test-images\GF2_PMS2_E114.0_N41.8_20230706_L1A0007380646\GF2_PMS2_E114.0_N41.8_20230706_L1A0007380646.tif")
# output_folder = r"F:\Test-images\GF2_PMS2_E114.0_N41.8_20230706_L1A0007380646\images"
# split_image(input_image_path, output_folder)



# from osgeo import gdal, osr
#
#
# def get_lat_lon_direction(latitude, longitude):
#     lat_direction = "北纬" if latitude >= 0 else "南纬"
#     lon_direction = "东经" if longitude >= 0 else "西经"
#     return lat_direction, lon_direction
#
#
# def main():
#     # 打开影像文件
#     ds = gdal.Open(
#         r"F:\Test-images\GF2_PMS2_E114.0_N41.8_20230706_L1A0007380646\GF2_PMS2_E114.0_N41.8_20230706_L1A0007380646.tif")
#     # 获取影像的地理转换信息和投影坐标系统
#     gt = ds.GetGeoTransform()
#     proj = ds.GetProjection()
#     # 创建投影坐标系对象
#     spatial_ref = osr.SpatialReference()
#     spatial_ref.ImportFromWkt(proj)
#     # 创建经纬度坐标系对象
#     lat_lon_srs = spatial_ref.CloneGeogCS()
#     # 创建坐标转换对象
#     transform = osr.CoordinateTransformation(spatial_ref, lat_lon_srs)
#     # 左上角坐标
#     ulx, uly, _ = transform.TransformPoint(gt[0], gt[3])
#     # 右下角坐标
#     lrx, lry, _ = transform.TransformPoint(gt[0] + gt[1] * ds.RasterXSize,
#                                            gt[3] + gt[5] * ds.RasterYSize)
#     # 获取经纬度方向
#     lat_direction, lon_direction = get_lat_lon_direction(uly, ulx)
#
#     # 打印经纬度范围和方向
#     print(f"左上角经纬度：{lat_direction}{abs(uly):.4f}, {lon_direction}{abs(ulx):.4f}")
#     print(f"右下角经纬度：{lat_direction}{abs(lry):.4f}, {lon_direction}{abs(lrx):.4f}")
#
#     # 关闭影像文件
#     ds = None
#
#
# if __name__ == "__main__":
#     main()



