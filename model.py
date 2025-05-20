#导入必要的库
import torch.nn as nn

# CNN模型
class RockPaperScissorsModel(nn.Module):
    def __init__(self):
        super(RockPaperScissorsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # 假设输入图像大小为 224x224
        self.fc2 = nn.Linear(128, 3)  # 3 个类别：rock, paper, scissors

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x