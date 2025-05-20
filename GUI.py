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
                             QVBoxLayout, QHBoxLayout, QWidget, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from torch.serialization import add_safe_globals  # 添加安全全局变量所需


class RockPaperScissorsGame(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化游戏变量
        self.classes = ['paper', 'rock', 'scissors']
        self.user_choice = None
        self.computer_choice = None
        self.result = ""

        # 设置窗口标题和尺寸
        self.setWindowTitle("石头剪刀布游戏")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

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
        # 顶部摄像头显示区域
        self.camera_label = QLabel("摄像头未启动")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.camera_label)

        # 游戏区域布局
        game_layout = QHBoxLayout()

        # 用户选择区域
        user_area = QVBoxLayout()
        self.user_title = QLabel("你的选择")
        self.user_title.setAlignment(Qt.AlignCenter)
        self.user_choice_label = QLabel("请出拳...")
        self.user_choice_label.setAlignment(Qt.AlignCenter)
        self.user_choice_label.setMinimumSize(200, 200)
        user_area.addWidget(self.user_title)
        user_area.addWidget(self.user_choice_label)

        # 结果显示区域
        result_area = QVBoxLayout()
        self.result_label = QLabel("游戏未开始")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        result_area.addWidget(self.result_label)

        # 电脑选择区域
        computer_area = QVBoxLayout()
        self.computer_title = QLabel("电脑选择")
        self.computer_title.setAlignment(Qt.AlignCenter)
        self.computer_choice_label = QLabel("等待中...")
        self.computer_choice_label.setAlignment(Qt.AlignCenter)
        self.computer_choice_label.setMinimumSize(200, 200)
        computer_area.addWidget(self.computer_title)
        computer_area.addWidget(self.computer_choice_label)

        # 添加到游戏布局
        game_layout.addLayout(user_area)
        game_layout.addLayout(result_area)
        game_layout.addLayout(computer_area)
        self.main_layout.addLayout(game_layout)

        # 底部按钮区域
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始游戏")
        self.start_button.clicked.connect(self.start_game)
        self.quit_button = QPushButton("退出游戏")
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.quit_button)
        self.main_layout.addLayout(button_layout)

    def init_camera(self):
        """初始化摄像头并启动计时器"""
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 每30毫秒更新一帧

    def update_frame(self):
        """更新摄像头帧"""
        ret, frame = self.camera.read()
        if ret:
            self.current_frame = frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
            self.camera_label.setPixmap(QPixmap.fromImage(p))

    def load_computer_images(self):
        """加载电脑选择的图片"""
        self.computer_images = {}
        for choice in self.classes:
            label = QLabel(choice.capitalize())
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
            label.setMinimumSize(200, 200)
            self.computer_images[choice] = QPixmap()
            painter = QPainter(self.computer_images[choice])
            painter.begin(self.computer_images[choice])
            painter.fillRect(0, 0, 200, 200, Qt.white)
            painter.setFont(QFont("Arial", 16))
            painter.drawText(100, 100, choice.capitalize())
            painter.end()

    def load_model(self):
        """加载训练好的模型"""
        try:
            # 添加模型所需的所有安全全局变量（根据模型架构调整）
            add_safe_globals([
                models.ResNet, nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                nn.MaxPool2d, nn.Linear, nn.Softmax
            ])

            # 加载完整模型（非权重字典），禁用weights_only并指定CPU加载
            self.model = torch.load(
                'full_model.pth',
                map_location=torch.device('cpu'),
                weights_only=False
            )
            self.model.eval()  # 评估模式

            # 确保预处理与训练时一致（调整尺寸和归一化参数）
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 模型输入尺寸
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

        # 显示ROI（调试用）
        cv2.imshow("Hand ROI", roi)
        cv2.waitKey(1)

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

        # 获取用户手势
        self.user_choice = self.process_frame(self.current_frame)

        # 显示用户选择
        rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        user_qimage = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        user_pixmap = QPixmap.fromImage(user_qimage).scaled(200, 200, Qt.KeepAspectRatio)
        self.user_choice_label.setPixmap(user_pixmap)
        self.user_choice_label.setText("")

        # 电脑随机选择
        self.computer_choice = random.choice(self.classes)
        self.computer_choice_label.setPixmap(self.computer_images[self.computer_choice])

        # 判断胜负
        self.determine_winner()

    def determine_winner(self):
        """判断游戏结果"""
        if self.user_choice == self.computer_choice:
            self.result = "平局!"
        elif (self.user_choice == 'rock' and self.computer_choice == 'scissors') or \
                (self.user_choice == 'scissors' and self.computer_choice == 'paper') or \
                (self.user_choice == 'paper' and self.computer_choice == 'rock'):
            self.result = "你赢了!"
        else:
            self.result = "电脑赢了!"

        self.result_label.setText(f"结果: {self.result}\n"
                                  f"你出了: {self.user_choice}, 电脑出了: {self.computer_choice}")

    def closeEvent(self, event):
        """释放资源"""
        if hasattr(self, 'camera'):
            self.camera.release()
            cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RockPaperScissorsGame()
    window.show()
    sys.exit(app.exec_())