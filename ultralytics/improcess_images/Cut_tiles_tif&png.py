import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from osgeo import gdal
import numpy as np
from PIL import ImageEnhance, Image
from tqdm import tqdm

gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")


# ===================== 子任务：切片（过滤三个波段全负值） =====================
def process_tile(input_image_path, x, y, tile_size, tile_folder, compress=True):
    input_dataset = gdal.Open(input_image_path)
    tile_x, tile_y = x * tile_size, y * tile_size
    tile_data = input_dataset.ReadAsArray(tile_x, tile_y, tile_size, tile_size)

    # 处理单波段和多波段情况
    if tile_data.ndim == 2:
        max_val = tile_data.max()
    else:  # 多波段
        max_val = tile_data.max(axis=(1, 2)).max()  # 各波段最大值，再取整体最大

    # 如果三个波段全部 < 0，直接跳过
    if max_val < 100:
        input_dataset = None
        return None

    output_tile_path = os.path.join(
        tile_folder,
        f"{os.path.splitext(os.path.basename(input_image_path))[0]}_{x}_{y}.tif"
    )
    driver = gdal.GetDriverByName('GTiff')
    num_bands = input_dataset.RasterCount
    # # 创建 GeoTIFF 时启用压缩
    creation_options = []
    if compress:
        creation_options = ["COMPRESS=LZW", "TILED=YES"]
        output_dataset = driver.Create(
            output_tile_path, tile_size, tile_size, num_bands,
            gdal.GDT_Float32, options=creation_options
        )
    else:
        output_dataset = driver.Create(output_tile_path, tile_size, tile_size, num_bands, gdal.GDT_Float32)

    output_dataset.SetProjection(input_dataset.GetProjection())
    geotransform = list(input_dataset.GetGeoTransform())
    geotransform[0] += tile_x * geotransform[1]
    geotransform[3] += tile_y * geotransform[5]
    output_dataset.SetGeoTransform(geotransform)

    for band_index in range(num_bands):
        band_data = tile_data[band_index, :, :] if tile_data.ndim == 3 else tile_data
        output_band = output_dataset.GetRasterBand(band_index + 1)
        output_band.WriteArray(band_data)

    output_dataset = None
    input_dataset = None
    return output_tile_path


# ===================== 子任务：TIF -> RGB PNG =====================
def convert_to_rgb(input_tif, output_image, contrast_factor=2.0):
    dataset = gdal.Open(input_tif)
    if dataset is None:
        raise RuntimeError(f"无法打开文件 {input_tif}")
    red = dataset.GetRasterBand(3).ReadAsArray().astype(np.float32)
    green = dataset.GetRasterBand(2).ReadAsArray().astype(np.float32)
    blue = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    red = (red / np.max(red)) * 255
    green = (green / np.max(green)) * 255
    blue = (blue / np.max(blue)) * 255
    rgb = np.dstack((red, green, blue))
    image = Image.fromarray(rgb.astype(np.uint8))
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image.save(output_image)


# ===================== 主流程 =====================
def process_image(input_image_path, tile_folder, png_folder=None,
                  tile_size=416, contrast_factor=2.0, to_png=True, max_workers=8):
    # 1. 计算任务总数
    dataset = gdal.Open(input_image_path)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    num_tiles_x, num_tiles_y = width // tile_size, height // tile_size
    total_tiles = num_tiles_x * num_tiles_y
    total_tasks = total_tiles + (total_tiles if to_png else 0)
    dataset = None

    if not os.path.exists(tile_folder):
        os.makedirs(tile_folder)
    if to_png and not os.path.exists(png_folder):
        os.makedirs(png_folder)

    tile_paths = []
    failed_png = []

    start_time = time.time()

    with tqdm(total=total_tasks, desc="总进度", unit="task") as pbar:

        # ========== 阶段 1：切片 ==========
        tasks = [(input_image_path, x, y, tile_size, tile_folder)
                 for y in range(num_tiles_y) for x in range(num_tiles_x)]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_tile, *args) for args in tasks]
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    tile_paths.append(result)
                pbar.update(1)

        # ========== 阶段 2：PNG 转换 ==========
        if to_png:
            def process_one(tif_file):
                try:
                    base_name = os.path.splitext(os.path.basename(tif_file))[0]
                    output_image = os.path.join(png_folder, f"{base_name}.png")
                    convert_to_rgb(tif_file, output_image, contrast_factor)
                    return True
                except Exception as e:
                    return (tif_file, str(e))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_one, tif) for tif in tile_paths]
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is not True:
                        failed_png.append(result)
                    pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time / 60:.2f} 分钟" if elapsed_time > 60 else f"\n总耗时: {elapsed_time:.2f} 秒")
    print(f"切片完成: {len(tile_paths)} 个")
    if to_png:
        print(f"PNG转换成功: {len(tile_paths) - len(failed_png)}, 失败: {len(failed_png)}")


# ===================== 使用示例 =====================
if __name__ == "__main__":
    input_image_path = r"F:\BaiduNetdiskDownload\GF2\tif\GF2_PMS1_E116.2_N44.0_20250511_L1A14627540001_fuse.tif"
    tile_folder = r"C:\Users\liuku\Desktop\GF2\GF2_PMS1_E116.2_N44.0_20250511_L1A14627540001_fuse_tiles"
    png_folder = r"C:\Users\liuku\Desktop\GF2_PMS1_E116.2_N44.0_20250511_L1A14627540001_fuse_png"

    process_image(input_image_path, tile_folder, png_folder,
                  tile_size=416, contrast_factor=2.0, to_png=True, max_workers=16)
