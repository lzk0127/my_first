import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# 配置参数
MODEL_PATH = 'full_model.pth'  # 训练好的完整模型路径
CLASS_NAMES = ['paper', 'rock', 'scissors']  # 类别名称（需与训练时一致）
CLASS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 每个类别的颜色(BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)  # 白色文本
BG_COLOR = (50, 50, 50)  # 背景色
BAR_WIDTH = 200  # 条形图宽度
BAR_HEIGHT = 30  # 每个条形图高度
BAR_SPACING = 10  # 条形图之间的间距
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, weights_only=False)
model.eval()  # 设置为评估模式

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(frame):
    """对输入帧进行预测并返回结果"""
    # 转换为PIL图像并预处理
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # 获取最高置信度的类别和值
    confidences, predicted_class = torch.topk(probabilities, 1)
    confidences = confidences.cpu().numpy()[0]
    class_idx = predicted_class.cpu().numpy()[0]

    return CLASS_NAMES[class_idx], confidences, probabilities.cpu().numpy()


def draw_result(frame, class_name, confidence, all_probabilities):
    """在图像上绘制分类结果和置信度条形图"""
    height, width, _ = frame.shape
    bar_area_height = (BAR_HEIGHT + BAR_SPACING) * len(CLASS_NAMES) + 50

    # 创建一个半透明的覆盖层用于显示条形图
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - bar_area_height - 10),
                  (width - 10, height - 10), BG_COLOR, -1)
    alpha = 0.8  # 透明度
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 绘制标题
    title = f"Prediction: {class_name} (Confidence: {confidence:.2f})"
    cv2.putText(frame, title, (20, height - bar_area_height + 25),
                FONT, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)

    # 绘制每个类别的置信度条形图
    y_pos = height - bar_area_height + 60
    for i, (name, prob, color) in enumerate(zip(CLASS_NAMES, all_probabilities, CLASS_COLORS)):
        # 绘制类别名称
        cv2.putText(frame, f"{name}:", (20, y_pos + 20),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # 计算条形图长度
        bar_length = int(prob * (width - 60))

        # 绘制背景条
        cv2.rectangle(frame, (100, y_pos),
                      (width - 20, y_pos + BAR_HEIGHT),
                      (100, 100, 100), -1)

        # 绘制置信度条
        cv2.rectangle(frame, (100, y_pos),
                      (100 + bar_length, y_pos + BAR_HEIGHT),
                      color, -1)

        # 绘制置信度值
        cv2.putText(frame, f"{prob:.2f}", (110, y_pos + 20),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        y_pos += BAR_HEIGHT + BAR_SPACING

    return frame


def main():
    cap = cv2.VideoCapture(0)  # 打开默认摄像头（0表示第一个摄像头）

    # 获取摄像头的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个窗口
    cv2.namedWindow('Real-time Classification', cv2.WINDOW_NORMAL)

    # 初始化上一帧的置信度，用于平滑过渡
    prev_probabilities = np.zeros(len(CLASS_NAMES))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预测结果
        class_name, confidence, all_probabilities = predict(frame)

        # 平滑过渡效果
        smooth_factor = 0.3
        all_probabilities = smooth_factor * all_probabilities + (1 - smooth_factor) * prev_probabilities
        prev_probabilities = all_probabilities.copy()

        # 绘制结果
        frame = draw_result(frame, class_name, confidence, all_probabilities)

        # 显示画面
        cv2.imshow('Real-time Classification', frame)

        # 按Q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()