# coding: utf-8
import os
import shutil

file1_folder = r'G:\wanxingyu\project\yolov10\yolov10-fuse\runs\detect\val-config-default\errorfile\FP'
file2_folder = r'G:\wanxingyu\project\yolov10\yolov10-fuse\runs\detect\predict2'
file3_folder = r'C:\Users\dell\Desktop\temp\temp1'

if not os.path.exists(file3_folder):
    os.makedirs(file3_folder)

for file_name in os.listdir(file1_folder):

    file1_path = os.path.join(file1_folder, file_name)

    if os.path.isfile(file1_path):
        file_name = file_name.replace('.txt', '.tif')
        file2_path = os.path.join(file2_folder, file_name)

        if os.path.isfile(file2_path):
            file3_path = os.path.join(file3_folder, file_name)
            shutil.copy(file2_path,file3_path)
            print(f'copy file {file_name} to {file3_folder}')
            # print(f"wo{file_name}")
        else:
            print(f'在file2中没有找到文件：{file_name}')
