import os
import glob
import math
import numpy as np
import matplotlib

# === 关键修改 1: 强制使用非交互式后端，防止 PyCharm 报错 ===
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# === 配置 ===
THUMB_SIZE = 256  # 缩略图读取大小
GRID_SIZE = 9  # 9x9 布局
IMAGES_PER_PAGE = GRID_SIZE * GRID_SIZE  # 每页 81 张


def linear_stretch(data):
    """
    2%-98% 线性拉伸，并强制裁剪到 0-1 之间，消除 Clipping 警告
    """
    data = np.nan_to_num(data)
    if np.max(data) == np.min(data):
        return data

    p2, p98 = np.percentile(data, (2, 98))
    img_clip = np.clip(data, p2, p98)
    img_norm = (img_clip - p2) / (p98 - p2)

    # === 关键修改 2: 强制 Clip 消除 Matplotlib 警告 ===
    return np.clip(img_norm, 0, 1)


def read_gf2_rgb(tif_path):
    ds = gdal.Open(tif_path)
    if not ds: return None
    w, h = ds.RasterXSize, ds.RasterYSize
    try:
        # GF-2: Band 3(R), 2(G), 1(B)
        r = ds.GetRasterBand(3).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        g = ds.GetRasterBand(2).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        b = ds.GetRasterBand(1).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        return np.dstack((linear_stretch(r), linear_stretch(g), linear_stretch(b)))
    except:
        return None


def read_gf3_gray(tif_path):
    ds = gdal.Open(tif_path)
    if not ds: return None
    w, h = ds.RasterXSize, ds.RasterYSize
    try:
        # GF-3: Band 1
        band1 = ds.GetRasterBand(1).ReadAsArray(0, 0, w, h, THUMB_SIZE, THUMB_SIZE)
        return linear_stretch(band1)
    except:
        return None


def plot_paginated(folder_path, satellite_type="GF2", base_save_name="result"):
    tif_files = glob.glob(os.path.join(folder_path, "*.tif"))
    total_files = len(tif_files)

    if total_files == 0:
        print(f"[{satellite_type}] 未找到文件: {folder_path}")
        return

    # 计算需要几页
    total_pages = math.ceil(total_files / IMAGES_PER_PAGE)
    print(f"--- 正在处理 {satellite_type}，共 {total_files} 个文件，将分为 {total_pages} 页保存 ---")

    for page in range(total_pages):
        start_idx = page * IMAGES_PER_PAGE
        end_idx = min((page + 1) * IMAGES_PER_PAGE, total_files)
        current_batch = tif_files[start_idx:end_idx]

        # 创建画布 24x24英寸，保证文件名有空间
        fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(24, 24))
        axes = axes.flatten()

        print(f"  正在生成第 {page + 1}/{total_pages} 页 (文件 {start_idx + 1} - {end_idx})...")

        for i in range(IMAGES_PER_PAGE):
            ax = axes[i]

            # 如果当前页还有图片
            if i < len(current_batch):
                tif_path = current_batch[i]
                file_name = os.path.basename(tif_path).split('_')[2] + "_" + os.path.basename(tif_path).split('_')[3] + "_" + os.path.basename(tif_path).split('_')[7] + "_" + os.path.basename(tif_path).split('_')[8]
                # 全名

                img_data = None
                if satellite_type == "GF2":
                    img_data = read_gf2_rgb(tif_path)
                elif satellite_type == "GF3":
                    img_data = read_gf3_gray(tif_path)

                if img_data is not None:
                    if satellite_type == "GF3":
                        ax.imshow(img_data, cmap='gray', vmin=0, vmax=1)
                    else:
                        ax.imshow(img_data)

                    # === 关键修改 3: 显示全名，调整字体和换行 ===
                    # fontsize=7 比较小，可以显示较长文件名
                    ax.set_title(file_name, fontsize=7, wrap=True)
                else:
                    ax.text(0.5, 0.5, "Read Error", ha='center')

            ax.axis('off')  # 关掉坐标轴

        plt.tight_layout()
        save_path = f"{base_save_name}_page_{page + 1}.png"
        plt.savefig(save_path, dpi=150)
        print(f"  ✅ 已保存: {save_path}")

        # 关闭当前图形，释放内存，防止 PyCharm 崩溃
        plt.close(fig)


if __name__ == "__main__":
    # 配置路径
    gf2_folder = r"C:\Users\liuku\Desktop\images"  # 你的路径
    gf3_folder = r"F:\my_code\TransmissionTower\images2\val\images"  # 你的路径

    # 执行 GF2 (自动分页)
    if os.path.exists(gf2_folder):
        plot_paginated(gf2_folder, "GF2", "GF2_Overview")

    # # 执行 GF3 (自动分页)
    # if os.path.exists(gf3_folder):
    #     plot_paginated(gf3_folder, "GF3", "GF3_Overview")

    print("--- 全部处理完成，请在文件夹中查看 png 图片 ---")
