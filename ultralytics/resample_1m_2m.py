import os
from osgeo import gdal
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import datetime


# === 日志锁（防止多线程写冲突） ===
log_lock = threading.Lock()

def log_message(log_file, message):
    """线程安全日志写入"""
    with log_lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")


def downsample_geotiff(in_path, out_path, scale=2.0, resample_alg='average', log_file=None):
    """单个 GeoTIFF 下采样"""
    try:
        ds = gdal.Open(in_path)
        if ds is None:
            raise Exception("无法打开文件")

        gt = ds.GetGeoTransform()
        pixel_width = gt[1]
        pixel_height = abs(gt[5])
        new_xres = pixel_width * scale
        new_yres = pixel_height * scale

        gdal.Warp(
            out_path, ds,
            xRes=new_xres,
            yRes=new_yres,
            resampleAlg=resample_alg,
            format='GTiff'
        )

        ds = None
        log_message(log_file, f"成功: {in_path}")
        return True

    except Exception as e:
        log_message(log_file, f"失败: {in_path} | 错误: {e}")
        return False


def downsample_geotiff_folder(input_folder, output_folder, scale=2.0, resample_alg='average', log_file=None, max_workers=4):
    """文件夹内所有 GeoTIFF 并行下采样"""
    os.makedirs(output_folder, exist_ok=True)
    tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
    if not tif_files:
        log_message(log_file, f"空文件夹: {input_folder}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for fname in tif_files:
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)

            if os.path.exists(out_path):
                log_message(log_file, f"跳过(已存在): {out_path}")
                continue

            futures.append(executor.submit(downsample_geotiff, in_path, out_path, scale, resample_alg, log_file))

        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"处理 {os.path.basename(input_folder)}", unit="file"):
            pass


def copy_label_folder(input_label_folder, output_label_folder, log_file=None):
    """复制 YOLO 标签文件"""
    os.makedirs(output_label_folder, exist_ok=True)
    label_files = [f for f in os.listdir(input_label_folder) if f.lower().endswith(".txt")]

    for fname in label_files:
        src = os.path.join(input_label_folder, fname)
        dst = os.path.join(output_label_folder, fname)
        try:
            shutil.copy(src, dst)
            log_message(log_file, f"拷贝标签: {fname}")
        except Exception as e:
            log_message(log_file, f"标签复制失败: {fname} | {e}")


def process_yolo_dataset(base_input_dir, base_output_dir, scale=2.0, resample_alg='average', max_workers=4):
    """批处理 YOLO GeoTIFF 数据集（带日志 & 多线程）"""
    log_file = os.path.join(base_output_dir, "process_log.txt")
    os.makedirs(base_output_dir, exist_ok=True)

    header = f"\n=== YOLO GeoTIFF 降分辨率日志 ===\n时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n输入: {base_input_dir}\n输出: {base_output_dir}\n比例: 1:{scale}\n算法: {resample_alg}\n线程: {max_workers}\n{'='*50}\n"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)

    subdirs = ["train", "val", "test"]
    for sub in subdirs:
        img_input = os.path.join(base_input_dir, "images", sub)
        lbl_input = os.path.join(base_input_dir, "labels", sub)

        img_output = os.path.join(base_output_dir, "images", sub)
        lbl_output = os.path.join(base_output_dir, "labels", sub)

        if os.path.exists(img_input):
            downsample_geotiff_folder(img_input, img_output, scale, resample_alg, log_file, max_workers)
        if os.path.exists(lbl_input):
            copy_label_folder(lbl_input, lbl_output, log_file)

    log_message(log_file, "全部处理完成！")
    print(f"\n处理完成，日志已保存：{log_file}")


if __name__ == "__main__":
    # === 参数设置 ===
    base_input_dir = r"D:\YOLO_dataset"         # 原始数据集根目录
    base_output_dir = r"D:\YOLO_dataset_2m"     # 输出数据集根目录
    scale = 2.0                                 # 分辨率比例 (1m → 2m)
    resample_alg = "average"                    # 重采样算法
    max_workers = 6                             # 线程数 (建议为 CPU 核数)

    process_yolo_dataset(base_input_dir, base_output_dir, scale, resample_alg, max_workers)
