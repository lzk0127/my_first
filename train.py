import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理和增强 - 增强版
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.25))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义模型 - 增加模型复杂度
def create_model(num_classes=3, dropout=0.5):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 冻结更少的层，只冻结前几层
    for param in list(model.parameters())[:10]:
        param.requires_grad = False

    # 修改分类器
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    return model


# 高级训练函数 - 支持混合精度和早停
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                num_epochs=25, early_stopping_patience=5, use_amp=True):
    best_acc = 0.0
    best_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping_counter = 0
    best_model_weights = None

    # 混合精度支持
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 收集预测结果用于分析
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 学习率调整
        scheduler.step(epoch_loss)

        # 早停机制
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = model.state_dict().copy()
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
            print(f'EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}')
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break

        # 计算每个类别的准确率
        class_correct = [0] * len(val_loader.dataset.classes)
        class_total = [0] * len(val_loader.dataset.classes)
        for i in range(len(val_labels)):
            label = val_labels[i]
            pred = val_preds[i]
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

        print("类别准确率:")
        for i in range(len(val_loader.dataset.classes)):
            if class_total[i] > 0:
                print(f'{val_loader.dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
            else:
                print(f'{val_loader.dataset.classes[i]}: N/A')

        epoch_time = time.time() - start_time
        print(f'Epoch time: {epoch_time:.2f}s')
        print()

    print(f'Best val Loss: {best_loss:4f}')

    # 加载最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, history


# 改进的训练历史可视化
def plot_training_history(history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


# 学习率查找器
def find_lr(model, criterion, optimizer, train_loader, init_value=1e-8, final_value=10.0, beta=0.98):
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    model.train()

    for inputs, labels in train_loader:
        batch_num += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 平滑损失计算
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # 记录最佳损失
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # 检测发散
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # 记录当前损失
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))

        # 反向传播
        loss.backward()
        optimizer.step()

        # 更新学习率
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    plt.figure(figsize=(10, 6))
    plt.plot(log_lrs, losses)
    plt.xlabel('Log LR')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.savefig('lr_finder.png')
    plt.show()

    # 找到最佳学习率
    idx_min = np.argmin(losses)
    idx_best = max(idx_min - 10, 0)  # 通常最佳LR在最小值之前
    best_lr = 10 ** log_lrs[idx_best]

    print(f"建议的学习率: {best_lr:.2e}")
    return best_lr


if __name__ == '__main__':
    # 确保在Windows上使用多进程时安全启动
    if hasattr(torch.multiprocessing, 'set_start_method'):
        torch.multiprocessing.set_start_method('spawn', force=True)

    # 数据集路径
    data_path = "split_data"

    # 创建数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=val_transform)

    # 打印类别信息
    print(f"训练集类别: {train_dataset.classes}")
    print(f"验证集类别: {val_dataset.classes}")

    # 计算类别权重以处理类别不平衡问题
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset:
        class_counts[label] += 1
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    print("类别权重:", class_weights)

    # 创建数据加载器 - 增加batch size
    batch_size = 32  # 增加batch size
    # 为Windows减少工作进程
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2 if os.name == 'nt' else 4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2 if os.name == 'nt' else 4, pin_memory=True)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")

    # 初始化模型
    model = create_model(len(train_dataset.classes)).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 定义损失函数和优化器
    # 使用类别权重的交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # 使用AdamW优化器替代Adam
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                              steps_per_epoch=len(train_loader),
                                              epochs=30, pct_start=0.1)

    # 学习率查找器（可选）
    # best_lr = find_lr(model, criterion, optimizer, train_loader)
    # optimizer = optim.AdamW(model.parameters(), lr=best_lr)

    # 训练模型
    print("开始训练模型...")
    model, history = train_model(
        model, criterion, optimizer, scheduler, train_loader, val_loader,
        num_epochs=30, early_stopping_patience=5, use_amp=True
    )

    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')
    # 保存完整模型
    torch.save(model, 'full_model.pth')

    # 显示训练历史
    plot_training_history(history)

    # 打印类别映射
    class_names = train_dataset.classes
    print("类别映射:")
    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")

    # 打印GPU内存使用情况（如果使用了GPU）
    if torch.cuda.is_available():
        print(f"训练后GPU内存占用: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")