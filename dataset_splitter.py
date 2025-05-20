import os
import shutil
import random
from sklearn.model_selection import train_test_split


def split_dataset(data_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    将数据集按指定比例划分为训练集、验证集和测试集

    参数:
    - data_dir: 原始数据集目录，包含多个类别文件夹
    - output_dir: 输出目录，将创建 train、val、test 子目录
    - train_ratio: 训练集比例 (默认0.6)
    - val_ratio: 验证集比例 (默认0.2)
    - test_ratio: 测试集比例 (默认0.2)
    - seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(seed)

    # 检查比例是否有效
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("训练集、验证集和测试集的比例之和必须为1.0")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有类别
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for class_name in classes:
        # 创建类别子目录
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 获取该类别的所有文件
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        if not files:
            print(f"警告: 类别 {class_name} 为空，跳过")
            continue

        # 划分数据集
        # 首先划分训练集和剩余数据
        train_files, remaining_files = train_test_split(
            files, test_size=val_ratio + test_ratio, random_state=seed
        )

        # 然后将剩余数据按 val_ratio:test_ratio 划分验证集和测试集
        val_files, test_files = train_test_split(
            remaining_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed
        )

        # 复制文件到对应的目录
        for file in train_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.copy2(src, dst)

        for file in val_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(val_dir, class_name, file)
            shutil.copy2(src, dst)

        for file in test_files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(test_dir, class_name, file)
            shutil.copy2(src, dst)

        print(f"类别 {class_name} 划分完成:")
        print(f"  训练集: {len(train_files)} 个样本")
        print(f"  验证集: {len(val_files)} 个样本")
        print(f"  测试集: {len(test_files)} 个样本")
        print(f"  总计: {len(files)} 个样本")
        print()

    total_train = sum(len(files) for _, _, files in os.walk(train_dir) if not files)
    total_val = sum(len(files) for _, _, files in os.walk(val_dir) if not files)
    total_test = sum(len(files) for _, _, files in os.walk(test_dir) if not files)

    print(f"数据集划分完成!")
    print(f"训练集总数: {total_train}")
    print(f"验证集总数: {total_val}")
    print(f"测试集总数: {total_test}")
    print(f"总样本数: {total_train + total_val + total_test}")


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"C:\Users\Administrator\Desktop\data"  # 原始数据集目录
    OUTPUT_DIR = "split_data"  # 输出目录

    # 执行划分
    split_dataset(DATA_DIR, OUTPUT_DIR)