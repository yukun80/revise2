import os
import random
import yaml
from pathlib import Path


def check_modality_files(data_root, file_id, modalities, check_label=True):
    """检查所有模态文件是否存在"""
    for modality in modalities:
        file_path = os.path.join(data_root, modality, f"{file_id}.tif")
        if not os.path.exists(file_path):
            return False

    if check_label:
        label_path = os.path.join(data_root, "label", f"{file_id}.tif")
        if not os.path.exists(label_path):
            return False

    return True


def generate_dataset_lists(config, train_ratio=0.5, seed=3407):
    """生成训练集和验证集列表

    Args:
        config: 配置文件字典
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)

    # 获取数据路径
    data_root = config["data"]["root_dir"]
    modalities = [k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")]

    # 获取所有文件ID
    all_files = []
    # 使用任意一个模态目录来获取文件列表
    modality_dir = os.path.join(data_root, modalities[0])
    for file_name in os.listdir(modality_dir):
        if file_name.endswith(".tif"):
            file_id = os.path.splitext(file_name)[0]
            # 检查所有模态文件是否都存在
            if check_modality_files(data_root, file_id, modalities):
                all_files.append(file_id)

    # 随机划分训练集和验证集
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # 保存训练集列表
    train_list_path = os.path.join(data_root, "train.txt")
    with open(train_list_path, "w") as f:
        for file_id in train_files:
            f.write(f"{file_id}\n")

    # 保存验证集列表
    val_list_path = os.path.join(data_root, "val.txt")
    with open(val_list_path, "w") as f:
        for file_id in val_files:
            f.write(f"{file_id}\n")

    print(f"Dataset split complete:")
    print(f"Total files: {len(all_files)}")
    print(f"Training set: {len(train_files)}")
    print(f"Validation set: {len(val_files)}")
    print(f"Lists saved to: {data_root}")


def main():
    # 注意：这里需要修改为当前的配置文件
    with open("configs/exp_swin-HARMF_250524.yml", "r") as f:
        config = yaml.safe_load(f)

    # 生成数据集列表
    generate_dataset_lists(config)


if __name__ == "__main__":
    main()
