# -*- coding: utf-8 -*-
# @Time    : 2025/11/5 20:43
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : FusionNN.py
# @Software: PyCharm

"""
Describe:
"""
import os
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 让当前目录优先
from ultralytics import YOLO
# 避免 MKL 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 通用路径解决方案 ===
def get_base_dir():
    """返回程序运行的真实路径（无论是脚本还是打包后的exe）"""
    if getattr(sys, 'frozen', False):  # 打包后的exe
        return os.path.dirname(sys.executable)
    else:  # 普通python脚本
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
# =====================


def read_config(xml_path="config.xml"):
    """读取XML配置文件并支持相对/绝对路径"""
    try:
        xml_path = os.path.join(BASE_DIR, xml_path)
        print("正在加载配置文件:", xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        device = root.findtext("device", default="0")
        input_folder = root.findtext("input_folder", default="testimages")
        output_folder = root.findtext("output_folder", default="runs/detect/pre")

        # 拼接到程序所在路径
        input_folder = os.path.join(BASE_DIR, input_folder)
        output_folder = os.path.join(BASE_DIR, output_folder)

        print(f"device: {device}")
        print(f"输入路径: {input_folder}")
        print(f"输出路径: {output_folder}")

        return device, input_folder, output_folder
    except Exception as e:
        print(f"配置文件读取失败: {e}")
        input("按回车键退出...")
        exit(1)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"已创建输出目录: {path}")


def run_detection():
    device, input_folder, output_folder = read_config()

    if not os.path.exists(input_folder):
        print(f"输入文件夹不存在: {input_folder}")
        input("按回车键退出...")
        exit(1)

    ensure_dir_exists(output_folder)

    model_path = os.path.join(BASE_DIR, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"权重文件未找到: {model_path}")
        input("按回车键退出...")
        exit(1)

    print("正在加载模型，请稍候...")
    model = YOLO(model_path)

    print("模型加载完成，开始预测...")
    model.predict(
        source=input_folder,
        imgsz=416,
        cache='disk',
        workers=0,
        device=device,
        exist_ok=True,
        save=True,
        visualize=False,
        name=output_folder
    )

    print(f"预测完成！结果已保存至: {output_folder}")
    input("按回车键退出程序...")


if __name__ == '__main__':
    run_detection()

