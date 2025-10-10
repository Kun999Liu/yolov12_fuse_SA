# coding: utf-8
import random

from ultralytics.utils.plotting import plot_images

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import os

# 绘制npy的网格和yolo格式的目标框

# npy文件的路径
current_folder = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\images\val'
npy_file = r'GF2_PMS1_E106.4_N37.4_20220326_L1A0006370789-pansharpencolNum_63rowNum_38.npy'
# 包含npy对应的label文件对应的文件夹
label_folder = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\labels\val'
# 将可视化后的文件保存到的文件路径
dist = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\nocut'
# 保存覆盖后的文件夹路径，不包含网格及标签框
final_dist = r'G:\wanxingyu\project\yolov10\yolov10-fuse\datasets\mydata-noCut\coverCut-val'



# 指定需要覆盖的编号块
target_grid_numbers = {3}

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
    '''
    获取YOLO框在图像上中对应的网格编号
    '''
    grid_numbers = set()
    for x in range(int(x_min),int(x_max)+1):
        for y in range(int(y_min),int(y_max)+1):
            grid_numbers.add(get_grid_number(x,y))
    return grid_numbers



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
    grid_numbers = {}
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # 计算网格编号
            grid_number = i*GRID_SIZE+j
            x = j*GRID_WIDTH+GRID_WIDTH//4
            y = i*GRID_HEIGHT+GRID_HEIGHT//4
            grid_numbers[grid_number] = (x, y)
            draw.text((x,y),str(grid_number),fill='white',font=font)



    # 读取txt
    # 存放归一化框对应的编号块
    occupied_grid_numbers = set()
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

            # 获取框所在的网格编号
            box_grid_numbers = get_box_grid_number(x_min,y_min,x_max,y_max,image_width,image_height)
            occupied_grid_numbers.update(box_grid_numbers)

            # draw.rectangle([x_min, y_min, x_max, y_max], width=2)



    # 找到所有不被框占据的网格编号
    availabel_grid_numbers = set(range(GRID_SIZE*GRID_SIZE)) - occupied_grid_numbers
    # 随机选择非占据编号来覆盖指定编号块
    availabel_grid_numbers = list(availabel_grid_numbers)
    random.shuffle(availabel_grid_numbers)

    # 替换编号块
    for target_number in target_grid_numbers:
        if availabel_grid_numbers:
            new_number = availabel_grid_numbers.pop()
            # 获取新编号的坐标位置
            x, y = grid_numbers[new_number]
            # 覆盖原来编号的位置
            draw.text((x,y),str(new_number),fill='white',font=font)

    jpg_file = os.path.splitext(npy_file)[0]+'.jpg'
    jpg_path = os.path.join(final_dist, jpg_file)
    image.save(jpg_path)
    print(f'Saved {jpg_file} to {final_dist}')



# for npy_file in npy_files:
#
#     image = np.load(npy_file)
#     plot_images(image, )