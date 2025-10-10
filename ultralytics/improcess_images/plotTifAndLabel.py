# coding: utf-8

from ultralytics.utils.plotting import plot_images

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import os

# 绘制npy的网格和yolo格式的目标框

# 包含npy文件的文件夹
current_folder = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\images\test'
# 包含npy对应的label文件对应的文件夹
label_folder = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\labels\test'
# 将可视化后的文件保存到的文件路径
dist = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\test-jpg'

# npy_files = glob.glob(os.path.join(current_folder, '*.npy'))
#
# print(npy_files)

GRID_SIZE = 4
GRID_WIDTH = 416 // GRID_SIZE
GRID_HEIGHT = 416 // GRID_SIZE

# 计算给定点（x，y）在网格中的编号
def get_grid_number(x,y):
    grid_x = x//GRID_WIDTH
    grid_y = y//GRID_HEIGHT
    return grid_y * GRID_SIZE + grid_x


# 获取框的归一化坐标所在的网格编号
def get_box_grid_number(x_min,y_min,x_max,y_max,image_width,image_height):
    grid_numbers = set()
    for x in range(int(x_min),int(x_max)+1):
        for y in range(int(y_min),int(y_max)+1):
            grid_numbers.add(get_grid_number(x,y))


for npy_file in os.listdir(current_folder):
    if npy_file.endswith('.npy'):
        # 获取对应的txt路径
        txt_file = os.path.splitext(npy_file)[0] + '.txt'
        txt_path = os.path.join(label_folder, txt_file)
        if os.path.exists(dist):
            # 读取.npy文件
            npy_path = os.path.join(current_folder, npy_file)
            image_data = np.load(npy_path)
            image_height, image_width, _ = image_data.shape

            image_data = image_data[...,:3]
            image_RGB = image_data[...,::-1]
            # 将.npy转换为图像
            image = Image.fromarray(image_RGB)

            # 准备绘制
            draw = ImageDraw.Draw(image)

            # 绘制网格线
            for i in range(1,GRID_SIZE):
                # 垂直网格线
                draw.line([i*GRID_WIDTH, 0, i*GRID_WIDTH, image_height],fill='white', width=2)
                # 水平网格线
                draw.line([0, i * GRID_HEIGHT, image_width, i * GRID_HEIGHT], fill='white', width=2)

            # 编号网络
            font = ImageFont.load_default()
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    # 计算网格编号
                    grid_number = i*GRID_SIZE+j
                    x = j*GRID_WIDTH+GRID_WIDTH//4
                    y = i*GRID_HEIGHT+GRID_HEIGHT//4
                    draw.text((x,y),str(grid_number),fill='white',font=font)


            # 读取txt
            with open(txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    # 将yolo的归一化坐标转换为像素坐标
                    x_min = (x_center - w / 2) * image_width
                    y_min = (y_center - h / 2) * image_height
                    x_max = (x_center + w / 2) * image_width
                    y_max = (y_center + h / 2) * image_height

                    draw.rectangle([x_min, y_min, x_max, y_max], width=2)
            jpg_file = os.path.splitext(npy_file)[0]+'.jpg'
            jpg_path = os.path.join(dist,jpg_file)
            image.save(jpg_path)
            print(f'Saved {jpg_file} to {dist}')



# for npy_file in npy_files:
#
#     image = np.load(npy_file)
#     plot_images(image, )