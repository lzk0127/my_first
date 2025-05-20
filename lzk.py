import sys
import cv2
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFrame)
from PyQt5.QtCore import QTimer, Qt, QSize, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QBrush, QPen


class AnimatedLabel(QLabel):
    """支持动画效果的自定义标签"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._opacity = 1.0

    def getOpacity(self):
        return self._opacity

    def setOpacity(self, opacity):
        self._opacity = opacity
        self.update()

    opacity = pyqtProperty(float, getOpacity, setOpacity)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(self.opacity)
        super().paintEvent(event)


class RockPaperScissorsGame(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化游戏变量
        self.classes = ['paper', 'rock', 'scissors']
        self.user_choice = None
        self.computer_choice = None
        self.result = ""

        # 颜色配置
        self.colors = {
            'primary': '#4361EE',
            'secondary': '#3A0CA3',
            'accent': '#F72585',
            'neutral': '#2B2D42',
            'neutral_light': '#8D99AE',
            'neutral_lighter': '#EDF2F4'
        }

        # 设置窗口标题和尺寸（缩小为紧凑布局）
        self.setWindowTitle("石头剪刀布游戏")
        self.setGeometry(100, 100, 640, 520)  # 缩小为 640x520
        self.setStyleSheet(f"background-color: {self.colors['neutral_lighter']};")

        # 创建中央部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # 创建游戏界面组件
        self.create_game_interface()

        # 初始化摄像头
        self.init_camera()

        # 加载电脑选择的图片
        self.load_computer_images()

        # 加载训练好的模型
        self.load_model()

    def create_game_interface(self):
        """创建游戏界面组件"""
        # 标题区域（缩小尺寸）
        title_frame = QFrame(self)
        title_frame.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.colors['primary']}, stop:1 {self.colors['secondary']});
            border-radius: 10px;
            padding: 10px;
        """)
        title_layout = QVBoxLayout(title_frame)

        title_label = QLabel("石头剪刀布")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        title_layout.addWidget(title_label)

        subtitle_label = QLabel("摄像头手势对战")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px;")
        title_layout.addWidget(subtitle_label)

        self.main_layout.addWidget(title_frame)

        # 摄像头显示区域（缩小尺寸）
        camera_frame = QFrame(self)
        camera_frame.setStyleSheet(f"""
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            padding: 10px;
        """)
        camera_layout = QVBoxLayout(camera_frame)

        self.camera_label = QLabel("摄像头未启动")
        self.camera_label.setMinimumSize(320, 240)  # 缩小为 320x240
        self.camera_label.setStyleSheet(
            "background-color: #f0f2f5; border-radius: 8px; color: #8D99AE; font-size: 14px;")
        camera_layout.addWidget(self.camera_label)

        self.gesture_indicator = QLabel("请出拳...")
        self.gesture_indicator.setAlignment(Qt.AlignCenter)
        self.gesture_indicator.setStyleSheet(f"""
            color: white;
            font-size: 14px;
            background-color: rgba(67,97,238,0.8);
            border-radius: 6px;
            padding: 4px;
            margin-top: 8px;
        """)
        camera_layout.addWidget(self.gesture_indicator)

        self.main_layout.addWidget(camera_frame)

        # 游戏区域布局（横向紧凑排列）
        game_frame = QFrame(self)
        game_frame.setStyleSheet(f"""
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            padding: 15px;
        """)
        game_layout = QHBoxLayout(game_frame)
        game_layout.setSpacing(10)

        # 用户选择区域（缩小尺寸）
        user_area = QVBoxLayout()

        self.user_title = QLabel("你")
        self.user_title.setStyleSheet(f"color: {self.colors['primary']}; font-size: 14px; font-weight: bold;")
        user_area.addWidget(self.user_title, alignment=Qt.AlignCenter)

        self.user_choice_label = AnimatedLabel()
        self.user_choice_label.setFixedSize(100, 100)  # 缩小为 100x100
        self.user_choice_label.setStyleSheet(
            f"background-color: {self.colors['neutral_lighter']}; border-radius: 10px;")
        user_area.addWidget(self.user_choice_label, alignment=Qt.AlignCenter)

        game_layout.addLayout(user_area)

        # 结果显示区域（居中紧凑）
        result_area = QVBoxLayout()

        self.result_label = QLabel("游戏未开始")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 12px;")
        result_area.addWidget(self.result_label)

        self.outcome_frame = QFrame()
        self.outcome_frame.setFixedSize(80, 80)  # 缩小为 80x80
        self.outcome_frame.setStyleSheet(f"""
            border-radius: 50%;
            background: {self.colors['neutral_lighter']};
            color: {self.colors['neutral_light']};
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
        """)
        outcome_layout = QVBoxLayout(self.outcome_frame)

        self.outcome_icon = QLabel("")
        self.outcome_icon.setAlignment(Qt.AlignCenter)
        self.outcome_icon.setStyleSheet("font-size: 24px;")
        outcome_layout.addWidget(self.outcome_icon)

        self.outcome_text = QLabel("开始")
        self.outcome_text.setAlignment(Qt.AlignCenter)
        self.outcome_text.setStyleSheet("font-size: 12px;")
        outcome_layout.addWidget(self.outcome_text)

        result_area.addWidget(self.outcome_frame, alignment=Qt.AlignCenter)

        game_layout.addLayout(result_area)

        # 电脑选择区域（缩小尺寸）
        computer_area = QVBoxLayout()

        self.computer_title = QLabel("电脑")
        self.computer_title.setStyleSheet(f"color: {self.colors['secondary']}; font-size: 14px; font-weight: bold;")
        computer_area.addWidget(self.computer_title, alignment=Qt.AlignCenter)

        self.computer_choice_label = AnimatedLabel()
        self.computer_choice_label.setFixedSize(100, 100)  # 缩小为 100x100
        self.computer_choice_label.setStyleSheet(
            f"background-color: {self.colors['neutral_lighter']}; border-radius: 10px;")
        computer_area.addWidget(self.computer_choice_label, alignment=Qt.AlignCenter)

        game_layout.addLayout(computer_area)

        self.main_layout.addWidget(game_frame)

        # 底部按钮区域（缩小按钮尺寸）
        button_frame = QFrame(self)
        button_frame.setStyleSheet(f"""
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            padding: 10px;
        """)
        button_layout = QHBoxLayout(button_frame)
        button_layout.setSpacing(10)

        self.start_button = QPushButton("开始游戏")
        self.start_button.setFixedSize(120, 40)  # 缩小为 120x40
        self.start_button.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.colors['primary']}, stop:1 {self.colors['secondary']});
            color: white;
            border-radius: 20px;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(67,97,238,0.3);
        """)
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.clicked.connect(self.start_game)
        button_layout.addWidget(self.start_button)

        self.quit_button = QPushButton("退出游戏")
        self.quit_button.setFixedSize(120, 40)  # 缩小为 120x40
        self.quit_button.setStyleSheet(f"""
            background-color: {self.colors['neutral']};
            color: white;
            border-radius: 20px;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(43,45,66,0.3);
        """)
        self.quit_button.setCursor(Qt.PointingHandCursor)
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.quit_button)

        self.main_layout.addWidget(button_frame)

    def init_camera(self):
        """初始化摄像头并启动计时器"""
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 每30毫秒更新一帧

    def update_frame(self):
        """更新摄像头帧（缩小显示尺寸）"""
        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(320, 240, Qt.KeepAspectRatio)  # 显示为 320x240
            self.camera_label.setPixmap(QPixmap.fromImage(p))

            # 实时检测手势并更新指示器
            if self.model is not None and hasattr(self, 'current_frame'):
                gesture = self.process_frame(self.current_frame)
                self.gesture_indicator.setText(f"检测到: {gesture.capitalize()}")

    def load_computer_images(self):
        """加载电脑选择的图标（缩小尺寸）"""
        self.computer_images = {}
        for choice in self.classes:
            pixmap = QPixmap(100, 100)  # 图标尺寸缩小为 100x100
            pixmap.fill(QColor(self.colors['neutral_lighter']))

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setFont(QFont("FontAwesome", 24))  # 图标字体缩小
            painter.setPen(QColor(self.colors['secondary']))

            if choice == 'paper':
                painter.drawText(50, 65, chr(0xf256))  # fa-hand-paper-o
            elif choice == 'rock':
                painter.drawText(50, 65, chr(0xf255))  # fa-hand-rock-o
            elif choice == 'scissors':
                painter.drawText(50, 65, chr(0xf257))  # fa-hand-scissors-o

            painter.end()

            self.computer_images[choice] = pixmap

    def load_model(self):
        """加载训练好的模型"""
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([
                models.ResNet, nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                nn.MaxPool2d, nn.Linear, nn.Softmax
            ])

            self.model = torch.load(
                'full_model.pth',
                map_location=torch.device('cpu'),
                weights_only=False
            )
            self.model.eval()  # 评估模式

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet均值
                    std=[0.229, 0.224, 0.225]  # ImageNet标准差
                )
            ])
            print("模型加载成功")

        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None  # 加载失败时设为None

    def get_hand_roi(self, frame):
        """优化版手部区域检测（结合肤色检测和轮廓提取）"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 肤色阈值范围（可根据环境光线调整）
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # 创建肤色掩码并形态学处理
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 取最大轮廓作为手部
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            # 扩展边界框以包含更多上下文
            margin = int(max(w, h) * 0.2)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)

            return frame[y:y + h, x:x + w]

        # 未检测到肤色时返回中心区域
        h, w, _ = frame.shape
        margin = min(h, w) // 3
        return frame[h // 2 - margin:h // 2 + margin, w // 2 - margin:w // 2 + margin]

    def process_frame(self, frame):
        """处理帧并进行手势识别（含调试输出）"""
        if self.model is None:
            print("警告：模型未加载，使用随机选择")
            return random.choice(self.classes)

        # 提取手部ROI
        roi = self.get_hand_roi(frame)
        if roi is None or roi.size == 0:
            print("警告：未检测到有效手部区域，使用随机选择")
            return random.choice(self.classes)

        # 预处理图像
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            class_index = torch.argmax(probs).item()

            # 打印调试信息
            print(f"预测结果：{self.classes[class_index]}, 概率：{probs[0][class_index]:.2f}")

        return self.classes[class_index]

    def start_game(self):
        """开始游戏流程"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            print("警告：未获取到摄像头画面")
            return

        # 重置动画状态
        self.user_choice_label.opacity = 0.0
        self.computer_choice_label.opacity = 0.0

        # 显示动画
        user_anim = QPropertyAnimation(self.user_choice_label, b"opacity")
        user_anim.setDuration(500)
        user_anim.setStartValue(0.0)
        user_anim.setEndValue(1.0)
        user_anim.start()

        computer_anim = QPropertyAnimation(self.computer_choice_label, b"opacity")
        computer_anim.setDuration(500)
        computer_anim.setStartValue(0.0)
        computer_anim.setEndValue(1.0)
        computer_anim.start()

        # 获取用户手势
        self.user_choice = self.process_frame(self.current_frame)

        # 显示用户选择（带圆角裁剪）
        rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        user_qimage = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        user_pixmap = QPixmap.fromImage(user_qimage).scaled(100, 100, Qt.KeepAspectRatio)

        rounded_pixmap = QPixmap(100, 100)
        rounded_pixmap.fill(Qt.transparent)
        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(user_pixmap))
        painter.drawRoundedRect(0, 0, 100, 100, 10, 10)
        painter.end()

        self.user_choice_label.setPixmap(rounded_pixmap)
        self.user_choice_label.setText("")

        # 电脑随机选择
        self.computer_choice = random.choice(self.classes)
        self.computer_choice_label.setPixmap(self.computer_images[self.computer_choice])
        self.computer_choice_label.setText("")

        # 判断胜负
        self.determine_winner()

    def determine_winner(self):
        """判断游戏结果"""
        if self.user_choice == self.computer_choice:
            self.result = "平局!"
            self.outcome_text.setText("平局")
            self.outcome_icon.setText("")
            self.outcome_frame.setStyleSheet(f"""
                border-radius: 50%;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.colors['neutral_light']}, stop:1 {self.colors['neutral']});
                color: white;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
            """)
        elif (self.user_choice == 'rock' and self.computer_choice == 'scissors') or \
                (self.user_choice == 'scissors' and self.computer_choice == 'paper') or \
                (self.user_choice == 'paper' and self.computer_choice == 'rock'):
            self.result = "你赢了!"
            self.outcome_text.setText("胜利")
            self.outcome_icon.setText(chr(0xf091))  # fa-trophy
            self.outcome_frame.setStyleSheet(f"""
                border-radius: 50%;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.colors['accent']}, stop:1 {self.colors['primary']});
                color: white;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
            """)
        else:
            self.result = "电脑赢了!"
            self.outcome_text.setText("失败")
            self.outcome_icon.setText(chr(0xf1f8))  # fa-meh-o
            self.outcome_frame.setStyleSheet(f"""
                border-radius: 50%;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.colors['neutral']}, stop:1 {self.colors['neutral_light']});
                color: white;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
            """)

        self.result_label.setText(f"结果: {self.result}\n"
                                  f"你出了: {self.user_choice}, 电脑出了: {self.computer_choice}")

        # 添加结果显示动画
        self.result_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 12px;
            color: {self.colors['primary']};
        """)

    def closeEvent(self, event):
        """释放资源"""
        if hasattr(self, 'camera'):
            self.camera.release()
            cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体以确保图标正确显示
    font = QFont()
    font.setFamily("SimHei")  # 设置中文字体
    app.setFont(font)

    window = RockPaperScissorsGame()
    window.show()
    sys.exit(app.exec_())