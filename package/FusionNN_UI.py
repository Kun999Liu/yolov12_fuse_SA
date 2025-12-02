# -*- coding: utf-8 -*-
"""
YOLO 目标检测可视化界面
带有现代化 QSS 样式和实时结果预览
"""
import ctypes
import os
import sys
import xml.etree.ElementTree as ET
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QComboBox, QProgressBar,
                             QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap
from ultralytics import YOLO

# 避免 MKL 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_base_dir():
    """返回程序运行的真实路径"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()


class DetectionThread(QThread):
    """检测线程，避免阻塞主界面"""
    progress = pyqtSignal(str)  # 进度信息
    progress_value = pyqtSignal(int, int)  # 当前进度和总数
    result_image = pyqtSignal(str)  # 结果图像路径
    finished = pyqtSignal(dict)  # 完成信号，返回统计结果
    error = pyqtSignal(str)  # 错误信号

    def __init__(self, model_path, input_folder, output_folder, device, imgsz):
        super().__init__()
        self.model_path = model_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = device
        self.imgsz = imgsz

    def run(self):
        try:
            self.progress.emit("正在加载模型...")
            model = YOLO(self.model_path)

            self.progress.emit("模型加载完成，开始预测...")

            # 确保输出目录存在
            os.makedirs(self.output_folder, exist_ok=True)

            # 解析项目路径和名称，确保保存路径完全匹配 output_folder
            project_path = os.path.dirname(self.output_folder)
            name_path = os.path.basename(self.output_folder)

            # 如果 output_folder 是根目录下的一级目录，dirname可能是空
            if not project_path:
                project_path = "."

            results = model.predict(
                source=self.input_folder,
                imgsz=self.imgsz,
                # cache='disk',
                workers=0,
                device=self.device,
                exist_ok=True,
                save=True,
                visualize=False,
                project=os.path.dirname(self.output_folder),
                name=os.path.basename(self.output_folder)
            )

            # 统计信息
            total_objects = 0
            total_time_ms = 0.0

            for i, r in enumerate(results, start=1):
                objs = len(r.boxes)
                t = r.speed['preprocess'] + r.speed['inference'] + r.speed['postprocess']
                total_objects += objs
                total_time_ms += t

                # 发送进度
                self.progress_value.emit(i, len(results))
                self.progress.emit(f"处理第 {i}/{len(results)} 张图像...")

                # 获取结果图像路径
                # --- 核心修复逻辑：直接构建路径，不依赖 r.save_dir ---
                if hasattr(r, 'path'):
                    img_name = os.path.basename(r.path)
                    # 强制去我们设定的 output_folder 找文件
                    result_img_path = os.path.join(self.output_folder, img_name)

                    # 检查文件是否存在（防止扩展名变化，如 png 变 jpg）
                    if os.path.exists(result_img_path):
                        self.result_image.emit(result_img_path)
                    else:
                        # 尝试寻找扩展名变化的情况（YOLO有时会把所有图片存为jpg）
                        name_no_ext = os.path.splitext(img_name)[0]
                        found = False
                        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                            probe_path = os.path.join(self.output_folder, name_no_ext + ext)
                            if os.path.exists(probe_path):
                                self.result_image.emit(probe_path)
                                found = True
                                break
                        if not found:
                            self.progress.emit(f"未找到结果文件: {result_img_path}")

            stats = {
                'total_images': len(results),
                'total_objects': total_objects,
                'total_time': total_time_ms / 1000,
                'avg_time': total_time_ms / len(results) if results else 0
            }

            self.finished.emit(stats)

        except Exception as e:
            import traceback
            traceback.print_exc()  # 打印详细错误到控制台
            self.error.emit(str(e))


class YOLODetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.result_images = []  # 存储所有结果图像路径
        self.current_image_index = 0  # 当前显示的图像索引
        self.init_ui()
        self.load_config()
        self.apply_styles()

    def init_ui(self):
        self.setWindowTitle("风机目标检测系统")
        self.setGeometry(100, 100, 1200, 700)

        # ==================== 新增图标代码开始 ====================
        # 1. 设置窗口左上角图标
        icon_path = os.path.join(BASE_DIR, "icon.ico")  # 假设图标名为 icon.ico
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            # 如果找不到 icon.ico，尝试找 icon.png
            png_path = os.path.join(BASE_DIR, "icon.png")
            if os.path.exists(png_path):
                self.setWindowIcon(QIcon(png_path))

        # 2. 修复 Windows 任务栏图标变成 Python 默认图标的问题
        # Windows 默认会将 Python 脚本归类为同一个组，导致任务栏不显示自定义图标
        if sys.platform == 'win32':
            try:
                # 任意唯一的字符串 ID
                myappid = 'mycompany.yolo.detection.v1'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception:
                pass
        # ==================== 新增图标代码结束 ====================

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # 改为水平布局
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左侧布局（原有内容）
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        # 标题
        title = QLabel("风机目标检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        left_layout.addWidget(title)

        # 配置区域
        config_group = QGroupBox("配置参数")
        config_layout = QGridLayout()
        config_layout.setSpacing(10)

        # 设备选择
        config_layout.addWidget(QLabel("运行设备:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "0", "1", "2", "3"])
        config_layout.addWidget(self.device_combo, 0, 1)

        # 图像尺寸
        config_layout.addWidget(QLabel("图像尺寸:"), 0, 2)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "640", "1280"])
        self.imgsz_combo.setCurrentText("416")
        config_layout.addWidget(self.imgsz_combo, 0, 3)

        # 权重路径
        config_layout.addWidget(QLabel("模型权重:"), 1, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择模型权重文件...")
        config_layout.addWidget(self.model_path_edit, 1, 1, 1, 2)
        self.model_browse_btn = QPushButton("浏览")
        self.model_browse_btn.clicked.connect(self.browse_model)
        config_layout.addWidget(self.model_browse_btn, 1, 3)

        # 输入文件夹
        config_layout.addWidget(QLabel("输入文件夹:"), 2, 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("选择输入图像文件夹...")
        config_layout.addWidget(self.input_path_edit, 2, 1, 1, 2)
        self.input_browse_btn = QPushButton("浏览")
        self.input_browse_btn.clicked.connect(self.browse_input)
        config_layout.addWidget(self.input_browse_btn, 2, 3)

        # 输出文件夹
        config_layout.addWidget(QLabel("输出文件夹:"), 3, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("选择输出结果文件夹...")
        config_layout.addWidget(self.output_path_edit, 3, 1, 1, 2)
        self.output_browse_btn = QPushButton("浏览")
        self.output_browse_btn.clicked.connect(self.browse_output)
        config_layout.addWidget(self.output_browse_btn, 3, 3)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # 控制按钮区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        control_layout.addWidget(self.save_config_btn)

        self.open_output_btn = QPushButton("打开输出文件夹")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        control_layout.addWidget(self.open_output_btn)

        left_layout.addLayout(control_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)  # 不确定进度模式
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # 识别进度条（显示具体进度）
        progress_group = QGroupBox("识别进度")
        progress_layout = QVBoxLayout()

        self.detection_progress_bar = QProgressBar()
        self.detection_progress_bar.setTextVisible(True)
        self.detection_progress_bar.setMinimum(0)
        self.detection_progress_bar.setMaximum(100)
        self.detection_progress_bar.setValue(0)
        self.detection_progress_bar.setFormat("%p% - %v/%m 张")
        progress_layout.addWidget(self.detection_progress_bar)

        progress_group.setLayout(progress_layout)
        left_layout.addWidget(progress_group)

        # 日志输出区域
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)

        # 将左侧布局添加到主布局
        main_layout.addLayout(left_layout, 2)  # 权重为2

        # 右侧布局（结果图框）
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # 结果图框
        result_group = QGroupBox("检测结果预览")
        result_layout = QVBoxLayout()

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 400)
        self.result_label.setStyleSheet("""
            QLabel {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: #ecf0f1;
                color: #95a5a6;
                font-size: 14px;
            }
        """)
        self.result_label.setText("暂无检测结果\n\n开始检测后将在此显示结果图像")
        self.result_label.setScaledContents(False)

        result_layout.addWidget(self.result_label)

        # 图像切换按钮
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)

        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.image_info_label = QLabel("0 / 0")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("font-weight: bold; color: #667eea;")
        nav_layout.addWidget(self.image_info_label)

        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        result_layout.addLayout(nav_layout)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(10)

        stats_layout.addWidget(QLabel("处理图像:"), 0, 0)
        self.stats_images_label = QLabel("0 张")
        self.stats_images_label.setStyleSheet("font-weight: bold; color: #667eea;")
        stats_layout.addWidget(self.stats_images_label, 0, 1)

        stats_layout.addWidget(QLabel("检测目标:"), 1, 0)
        self.stats_objects_label = QLabel("0 个")
        self.stats_objects_label.setStyleSheet("font-weight: bold; color: #667eea;")
        stats_layout.addWidget(self.stats_objects_label, 1, 1)

        stats_layout.addWidget(QLabel("总耗时:"), 2, 0)
        self.stats_time_label = QLabel("0.00 秒")
        self.stats_time_label.setStyleSheet("font-weight: bold; color: #667eea;")
        stats_layout.addWidget(self.stats_time_label, 2, 1)

        stats_layout.addWidget(QLabel("平均速度:"), 3, 0)
        self.stats_speed_label = QLabel("0.00 ms/张")
        self.stats_speed_label.setStyleSheet("font-weight: bold; color: #667eea;")
        stats_layout.addWidget(self.stats_speed_label, 3, 1)

        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        right_layout.addStretch()

        # 将右侧布局添加到主布局
        main_layout.addLayout(right_layout, 1)  # 权重为1

        # 状态栏
        self.statusBar().showMessage("就绪")

    def apply_styles(self):
        """应用现代化 QSS 样式"""
        qss = """
        QMainWindow {
            background-color: #f5f5f5;
        }

        #titleLabel {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            padding: 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #667eea, stop:1 #764ba2);
            border-radius: 10px;
            color: white;
        }

        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            color: #34495e;
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 10px;
            background-color: white;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;
            color: #667eea;
        }

        QLabel {
            color: #2c3e50;
            font-size: 13px;
        }

        QLineEdit {
            padding: 8px;
            border: 2px solid #ecf0f1;
            border-radius: 5px;
            background-color: #ffffff;
            font-size: 13px;
            color: #2c3e50;
        }

        QLineEdit:focus {
            border: 2px solid #667eea;
        }

        QComboBox {
            padding: 8px;
            border: 2px solid #ecf0f1;
            border-radius: 5px;
            background-color: white;
            font-size: 13px;
            color: #2c3e50;
        }

        QComboBox:hover {
            border: 2px solid #667eea;
        }

        QComboBox::drop-down {
            border: none;
            padding-right: 10px;
        }

        QPushButton {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #667eea;
            color: white;
            font-size: 13px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #5568d3;
        }

        QPushButton:pressed {
            background-color: #4451b8;
        }

        #startButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #11998e, stop:1 #38ef7d);
            font-size: 15px;
            padding: 12px 30px;
        }

        #startButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #0e8577, stop:1 #2dd969);
        }

        QTextEdit {
            border: 2px solid #ecf0f1;
            border-radius: 5px;
            background-color: #2c3e50;
            color: #ecf0f1;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            padding: 10px;
        }

        QProgressBar {
            border: 2px solid #ecf0f1;
            border-radius: 5px;
            text-align: center;
            background-color: #ecf0f1;
            height: 25px;
        }

        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #667eea, stop:1 #764ba2);
            border-radius: 3px;
        }

        QStatusBar {
            background-color: #34495e;
            color: white;
            font-size: 12px;
        }
        """
        self.setStyleSheet(qss)

    def load_config(self):
        """加载配置文件"""
        try:
            xml_path = os.path.join(BASE_DIR, "config.xml")
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()

                device = root.findtext("device", default="0")
                input_folder = root.findtext("input_folder", default="testimages")
                output_folder = root.findtext("output_folder", default="runs/detect/pre")

                self.device_combo.setCurrentText(device)
                self.input_path_edit.setText(os.path.join(BASE_DIR, input_folder))
                self.output_path_edit.setText(os.path.join(BASE_DIR, output_folder))

                model_path = os.path.join(BASE_DIR, "weights", "best.pt")
                self.model_path_edit.setText(model_path)

                self.log("✓ 配置文件加载成功")
            else:
                # 默认配置
                self.model_path_edit.setText(os.path.join(BASE_DIR, "weights", "best.pt"))
                self.input_path_edit.setText(os.path.join(BASE_DIR, "testimages"))
                self.output_path_edit.setText(os.path.join(BASE_DIR, "runs"))
                self.log("未找到配置文件，使用默认配置")
        except Exception as e:
            self.log(f"配置加载失败: {e}")

    def save_config(self):
        """保存配置到 XML 文件"""
        try:
            root = ET.Element("config")
            ET.SubElement(root, "device").text = self.device_combo.currentText()

            # 保存相对路径
            input_rel = os.path.relpath(self.input_path_edit.text(), BASE_DIR)
            output_rel = os.path.relpath(self.output_path_edit.text(), BASE_DIR)

            ET.SubElement(root, "input_folder").text = input_rel
            ET.SubElement(root, "output_folder").text = output_rel

            tree = ET.ElementTree(root)
            xml_path = os.path.join(BASE_DIR, "config.xml")
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            self.log("配置已保存到 config.xml")
            QMessageBox.information(self, "成功", "配置保存成功！")
        except Exception as e:
            self.log(f"配置保存失败: {e}")
            QMessageBox.warning(self, "错误", f"配置保存失败：{e}")

    def browse_model(self):
        """选择模型文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重文件", BASE_DIR, "模型文件 (*.pt)")
        if path:
            self.model_path_edit.setText(path)

    def browse_input(self):
        """选择输入文件夹"""
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹", BASE_DIR)
        if path:
            self.input_path_edit.setText(path)

    def browse_output(self):
        """选择输出文件夹"""
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", BASE_DIR)
        if path:
            self.output_path_edit.setText(path)

    def open_output_folder(self):
        """打开输出文件夹"""
        output_path = self.output_path_edit.text()
        if os.path.exists(output_path):
            os.startfile(output_path)
        else:
            QMessageBox.warning(self, "提示", "输出文件夹不存在")

    def log(self, message):
        """添加日志"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def update_detection_progress(self, current, total):
        """更新检测进度条"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.detection_progress_bar.setMaximum(total)
            self.detection_progress_bar.setValue(current)
            self.detection_progress_bar.setFormat(f"{percentage}% - {current}/{total} 张")

    def display_result_image(self, image_path):
        """显示结果图像"""
        try:
            if image_path and os.path.exists(image_path):
                # 添加到结果图像列表
                if image_path not in self.result_images:
                    self.result_images.append(image_path)
                    self.current_image_index = len(self.result_images) - 1
                    self.show_current_image()
        except Exception as e:
            self.log(f"显示图像失败: {e}")

    def show_current_image(self):
        """显示当前索引的图像"""
        if not self.result_images:
            return

        try:
            image_path = self.result_images[self.current_image_index]
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 缩放图像以适应标签大小
                scaled_pixmap = pixmap.scaled(
                    self.result_label.width() - 10,
                    self.result_label.height() - 10,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.result_label.setPixmap(scaled_pixmap)

                # 更新图像信息
                self.image_info_label.setText(f"{self.current_image_index + 1} / {len(self.result_images)}")

                # 更新按钮状态
                self.prev_btn.setEnabled(self.current_image_index > 0)
                self.next_btn.setEnabled(self.current_image_index < len(self.result_images) - 1)
        except Exception as e:
            self.log(f"显示图像失败: {e}")

    def show_previous_image(self):
        """显示上一张图像"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def show_next_image(self):
        """显示下一张图像"""
        if self.current_image_index < len(self.result_images) - 1:
            self.current_image_index += 1
            self.show_current_image()

    def start_detection(self):
        """开始检测"""
        # 验证路径
        model_path = self.model_path_edit.text()
        input_folder = self.input_path_edit.text()
        output_folder = self.output_path_edit.text()

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "模型权重文件不存在！")
            return

        if not os.path.exists(input_folder):
            QMessageBox.warning(self, "错误", "输入文件夹不存在！")
            return

        # --- 关键修改：去除路径末尾的斜杠，防止 basename 获取为空 ---
        output_folder = output_folder.rstrip("\\").rstrip("/")
        # -------------------------------------------------------

        # 创建输出目录
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法创建输出目录: {e}")
            return

        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()

        # 重置统计信息和进度
        self.result_images = []  # 清空结果图像列表
        self.current_image_index = 0
        self.detection_progress_bar.setValue(0)
        self.stats_images_label.setText("0 张")
        self.stats_objects_label.setText("0 个")
        self.stats_time_label.setText("0.00 秒")
        self.stats_speed_label.setText("0.00 ms/张")
        self.result_label.clear()
        self.result_label.setText("检测中...\n\n请稍候")
        self.image_info_label.setText("0 / 0")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.log("=" * 50)
        self.log("开始检测任务...")
        self.log("=" * 50)

        # 创建检测线程
        device = self.device_combo.currentText()
        imgsz = int(self.imgsz_combo.currentText())

        self.detection_thread = DetectionThread(
            model_path, input_folder, output_folder, device, imgsz
        )
        self.detection_thread.progress.connect(self.log)
        self.detection_thread.progress_value.connect(self.update_detection_progress)
        self.detection_thread.result_image.connect(self.display_result_image)
        self.detection_thread.finished.connect(self.detection_finished)
        self.detection_thread.error.connect(self.detection_error)
        self.detection_thread.start()

    def detection_finished(self, stats):
        """检测完成"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        output_path = self.output_path_edit.text()

        # 更新统计信息
        self.stats_images_label.setText(f"{stats['total_images']} 张")
        self.stats_objects_label.setText(f"{stats['total_objects']} 个")
        self.stats_time_label.setText(f"{stats['total_time']:.2f} 秒")
        self.stats_speed_label.setText(f"{stats['avg_time']:.2f} ms/张")

        self.log("=" * 50)
        self.log("✓ 检测完成！")
        self.log("=" * 50)
        self.log(f"总图像数: {stats['total_images']}")
        self.log(f"检测目标数: {stats['total_objects']}")
        self.log(f"总耗时: {stats['total_time']:.3f} 秒")
        self.log(f"平均耗时: {stats['avg_time']:.2f} ms/图")
        self.log(f"结果保存至: {output_path}")
        self.log("=" * 50)

        self.statusBar().showMessage("检测完成！")

        # 询问是否打开输出文件夹
        reply = QMessageBox.question(self, "完成",
                                     f"检测完成！\n\n"
                                     f"处理图像: {stats['total_images']} 张\n"
                                     f"检测目标: {stats['total_objects']} 个\n"
                                     f"总耗时: {stats['total_time']:.2f} 秒\n\n"
                                     f"是否打开输出文件夹？",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.open_output_folder()

    def detection_error(self, error_msg):
        """检测出错"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log(f"✗ 错误: {error_msg}")
        self.statusBar().showMessage("检测失败")
        QMessageBox.critical(self, "错误", f"检测过程出错：\n{error_msg}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用 Fusion 风格作为基础

    window = YOLODetectionGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()