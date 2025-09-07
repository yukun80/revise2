import os
import yaml
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard支持

from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations_traditional import (
    get_traditional_train_augmentation,
    get_traditional_val_augmentation,
)
from utils.loss_factory import LossFactory
from models.model_factory import create_model  # Import the model factory
from utils.optimizer import PolyWarmupAdamW, CosineAnnealingWarmupAdamW
from utils.helpers import print_log
from utils.meter import AverageMeter, confusion_matrix, getScores  # 导入评估指标计算工具

import warnings
import logging

warnings.filterwarnings("ignore")  # 忽略警告


def parse_args():
    """
    解析命令行参数
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="本地进程ID",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="是否进行评估",
    )
    return parser.parse_args()


# 单机单卡：不需要分布式初始化
def setup_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        args.gpu = 0
    else:
        args.gpu = None
    return args


def create_dataloader(config, is_train=True):
    """创建数据加载器"""
    try:
        modalities = [
            k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")
        ]  # 获取所有启用的模态，并去除use_前缀，创建列表

        # 正确获取归一化配置
        norm_config = config["model"]["modality_norms"]

        # 构建文件列表的完整路径
        root_dir = config["data"]["root_dir"]
        file_list = os.path.join(root_dir, config["data"]["train_list"] if is_train else config["data"]["val_list"])

        # 创建数据集
        dataset = MultiModalRSDataset(
            root_dir=root_dir,
            file_list=file_list,  # 使用完整路径
            modalities=modalities,
            transform=(
                get_traditional_train_augmentation(norm_config)
                if is_train
                else get_traditional_val_augmentation(norm_config)
            ),
            stage="train" if is_train else "val",
        )

        # 检查数据集是否为空
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty!")

        # 创建数据加载器（单机单卡，无分布式采样器）
        sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(sampler is None and is_train),
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
            sampler=sampler,
        )

        return dataloader, sampler
    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise


def train_epoch(model, dataloader, optimizer, criterion, epoch, writer=None, print_freq=10):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    batch_loss = AverageMeter()

    with tqdm(total=len(dataloader)) as pbar:
        for i, (images, labels, meta) in enumerate(dataloader):
            images = [img.cuda() for img in images] if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # 获取模型输出
            outputs, aux_loss = model(images)

            # 计算主任务损失
            task_loss = criterion(outputs, labels)

            # 如果使用特征级融合，添加HMCB正则化损失
            use_feature_fusion = getattr(model, "use_feature_fusion", False)
            hmcb_weight = getattr(model, "hmcb_weight", 0.1)

            if use_feature_fusion and aux_loss is not None:
                total_loss_value = task_loss + hmcb_weight * aux_loss
            else:
                total_loss_value = task_loss

            # 优化
            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()

            # 更新指标
            batch_loss.update(total_loss_value.item())
            total_loss += total_loss_value.item()

            # 更新进度条
            if i % print_freq == 0:
                pbar.set_description(f"Epoch {epoch} Loss: {batch_loss.avg:.4f}")
            pbar.update(1)

            # 记录到TensorBoard
            if writer:
                global_step = epoch * len(dataloader) + i
                writer.add_scalar("Train/BatchLoss", total_loss_value.item(), global_step)
                writer.add_scalar("Train/TaskLoss", total_loss_value.item(), global_step)

                # 学习率记录
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Train/LearningRate", current_lr, global_step)

    # 记录每个epoch的平均损失
    if writer:
        writer.add_scalar("Train/EpochLoss", batch_loss.avg, epoch)

    return total_loss / len(dataloader)


class Saver:
    def __init__(self, args, ckpt_dir, best_val=0, condition=lambda x, y: x > y, save_interval=10):
        """初始化Saver

        Args:
            args: 配置参数
            ckpt_dir: 检查点保存目录
            best_val: 最佳验证指标
            condition: 比较函数，用于判断是否需要保存检查点
            save_interval: 定期保存的epoch间隔
        """
        self.args = args
        self.directory = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self.save_interval = save_interval

        # 创建保存目录
        os.makedirs(self.directory, exist_ok=True)

    def save(self, val_score, state_dict, epoch=None):
        """保存检查点

        Args:
            val_score: 验证指标 (IoU值)
            state_dict: 模型状态字典
            epoch: 当前训练轮次
        """
        # 保存最新的检查点
        latest_path = os.path.join(self.directory, "model_latest.pth")
        torch.save(state_dict, latest_path)

        # 如果验证指标更好，保存最佳模型
        if self.condition(val_score, self.best_val):
            best_path = os.path.join(self.directory, "model_best.pth")
            torch.save(state_dict, best_path)
            self.best_val = val_score

            # 保存最佳分数
            with open(os.path.join(self.directory, "best_score.txt"), "w") as f:
                f.write(f"Best IoU: {self.best_val:.4f}, Epoch: {epoch}")

            print_log(f"Saved new best model with IoU: {self.best_val:.4f}")

        # 周期性保存模型检查点
        if epoch is not None and (epoch + 1) % self.save_interval == 0:
            periodic_path = os.path.join(self.directory, f"model_epoch_{epoch+1}.pth")
            torch.save(state_dict, periodic_path)
            print_log(f"Saved periodic checkpoint at epoch {epoch}")


def validate(model, dataloader, criterion, epoch=0, writer=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    batch_loss = AverageMeter()

    # 初始化混淆矩阵
    conf_mat = np.zeros((2, 2))

    with torch.no_grad():
        for i, (images, labels, meta) in enumerate(dataloader):
            images = [img.cuda() for img in images] if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # 获取模型输出
            outputs, aux_loss = model(images)

            # 计算损失
            task_loss = criterion(outputs, labels)

            use_feature_fusion = getattr(model, "use_feature_fusion", False)
            hmcb_weight = getattr(model, "hmcb_weight", 0.1)

            if use_feature_fusion and aux_loss is not None:
                total_loss_value = task_loss + hmcb_weight * aux_loss
            else:
                total_loss_value = task_loss

            # 处理模型输出用于评估
            # 统一从列表或张量中取主输出
            if isinstance(outputs, list):
                ensemble_output = outputs[-1]
            else:
                ensemble_output = outputs

            # 更新损失指标
            batch_loss.update(total_loss_value.item())
            total_loss += total_loss_value.item()

            # 调整输出尺寸
            ensemble_output = nn.functional.interpolate(
                ensemble_output, size=labels.shape[1:], mode="bilinear", align_corners=False
            )

            try:
                # Properly detach tensors before converting to NumPy
                labels_np = labels.cpu().detach().numpy()
                predictions = torch.argmax(ensemble_output, dim=1).cpu().detach().numpy()

                # Ensure correct data types
                labels_np = labels_np.astype(np.int64)
                predictions = predictions.astype(np.int64)

                # 计算混淆矩阵
                batch_conf_mat = confusion_matrix(labels_np.flatten(), predictions.flatten(), 2, ignore_label=255)
                conf_mat += batch_conf_mat

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

    # 计算评估指标
    overall_acc, class_acc, iou = getScores(conf_mat)

    # 记录验证指标
    if writer:
        writer.add_scalar("Val/Loss", batch_loss.avg, epoch)
        writer.add_scalar("Val/Accuracy", overall_acc, epoch)
        writer.add_scalar("Val/mIoU", iou, epoch)
        writer.add_scalar("Val/ClassAccuracy", class_acc, epoch)

        # 记录每个类别的IoU
        class_names = ["Background", "Target"]
        for i, class_iou in enumerate(getIoUPerClass(conf_mat)):
            if i < len(class_names):
                writer.add_scalar(f"Val/IoU_{class_names[i]}", class_iou, epoch)

    print_log(f"Validation Epoch {epoch}: Loss={batch_loss.avg:.4f}, Acc={overall_acc:.2f}%, mIoU={iou:.2f}%")

    return total_loss / len(dataloader), iou


def getIoUPerClass(confusion_matrix):
    """计算每个类别的IoU"""
    iou_list = []

    for i in range(confusion_matrix.shape[0]):
        # 对角线上的值（真阳性）
        true_positive = confusion_matrix[i, i]
        # 行和列的和减去对角线元素（交并比的分母）
        union = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - true_positive

        # 避免除以零
        if union > 0:
            iou = true_positive / union * 100
        else:
            iou = 0

        iou_list.append(iou)

    return iou_list


def main():
    """
    主函数
    """
    args = parse_args()

    # 读取配置文件
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 获取当前时间，格式为 YYYYMMDD（例如 20231015）
    current_time = datetime.datetime.now().strftime("%Y%m%d")
    # 更新配置字典
    config["training"]["current_time"] = current_time
    # 将更新后的配置写回文件
    with open(args.config, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # 设置设备（单机单卡）
    args = setup_device(args)

    # 创建TensorBoard日志目录
    log_dir = os.path.join(
        config["training"].get("log_dir", "logs"), config["training"]["name"] + "_" + config["training"]["current_time"]
    )

    # 创建SummaryWriter
    writer = SummaryWriter(log_dir)
    print_log(f"TensorBoard logs will be saved to {log_dir}")

    # 创建模型
    model = create_model(config)
    if torch.cuda.is_available():
        model = model.cuda()

    # 创建数据加载器
    train_loader, train_sampler = create_dataloader(config, is_train=True)
    val_loader, _ = create_dataloader(config, is_train=False)

    # 创建优化器
    param_groups = model.get_param_groups()

    scheduler_type = config["training"].get("scheduler", {}).get("type", "polynomial")

    if scheduler_type.lower() == "cos":
        # Use Cosine Annealing scheduler
        optimizer = CosineAnnealingWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": config["training"]["learning_rate"],
                    "weight_decay": config["training"]["weight_decay"],
                },
                {"params": param_groups[1], "lr": config["training"]["learning_rate"], "weight_decay": 0.0},
                {
                    "params": param_groups[2],
                    "lr": config["training"]["learning_rate"] * 10,
                    "weight_decay": config["training"]["weight_decay"],
                },
            ],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            betas=[0.9, 0.999],
            T_max=int(config["training"]["scheduler"].get("T_max", 40000)),
            eta_min=float(config["training"]["scheduler"].get("eta_min", 1e-7)),
            warmup_iterations=int(config["training"]["scheduler"].get("warmup_iterations", 3000)),
            warmup_ratio=float(config["training"]["scheduler"].get("warmup_ratio", 1e-6)),
        )
    else:
        # Use default Polynomial scheduler
        optimizer = PolyWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": config["training"]["learning_rate"],
                    "weight_decay": config["training"]["weight_decay"],
                },
                {"params": param_groups[1], "lr": config["training"]["learning_rate"], "weight_decay": 0.0},
                {
                    "params": param_groups[2],
                    "lr": config["training"]["learning_rate"] * 10,
                    "weight_decay": config["training"]["weight_decay"],
                },
            ],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            betas=[0.9, 0.999],
            warmup_iter=int(config["training"].get("optimizer_params", {}).get("warmup_iter", 1500)),
            max_iter=int(config["training"].get("optimizer_params", {}).get("max_iter", 40000)),
            warmup_ratio=float(config["training"].get("optimizer_params", {}).get("warmup_ratio", 1e-6)),
            power=float(config["training"].get("optimizer_params", {}).get("power", 1.0)),
        )

    # 创建损失函数
    criterion = LossFactory.create_loss(config)

    # 创建保存器，设置每20个epoch保存一个检查点
    saver = Saver(
        args=config,
        ckpt_dir=os.path.join(
            config["training"]["ckpt_dir"], config["training"]["name"] + "_" + config["training"]["current_time"]
        ),
        best_val=0,
        condition=lambda x, y: x > y,  # IoU越高越好
        save_interval=10,  # 每20个epoch保存一个检查点
    )

    # 记录学习率
    if writer:
        writer.add_scalar("Train/LearningRate", config["training"]["learning_rate"], 0)

    # 训练循环
    for epoch in range(config["training"]["num_epochs"]):
        # 单机单卡，无需设置采样器 epoch

        # 训练一个epoch - 保持原始函数调用
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, writer)

        # 验证和保存
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            val_loss, val_iou = validate(model, val_loader, criterion, epoch, writer)
            print_log(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}%"
            )

            # 保存模型
            saver.save(
                val_iou,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                },
                epoch=epoch,
            )

    # 关闭TensorBoard writer
    if writer:
        writer.close()
    print_log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}%")
    # 最终保存模型
    saver.save(
        val_iou,
        {
            "model": model.state_dict(),
            "epoch": epoch,
        },
        epoch=epoch,
    )


if __name__ == "__main__":
    main()


"""
1. 单机单卡训练:
```bash
python train.py --config configs/exp_swin-HARMF_250524.yml
python train.py --config configs/exp_haefnet_test.yml
```

2. 分布式训练:
```bash
python -m torch.distributed.launch --nproc_per_node=N train.py --config configs/experiment.yml
```

3. 查看训练过程可视化:
   a. 安装TensorBoard（如果尚未安装）:
   ```bash
   pip install tensorboard
   ```
   
   b. 启动TensorBoard服务:
   ```bash
   tensorboard --logdir=logs
   ```
   
   c. 在浏览器中打开:
   http://localhost:6006
   
   在TensorBoard界面中，您可以查看:
   - Train/BatchLoss: 每个批次的训练损失
   - Train/EpochLoss: 每个Epoch的平均训练损失
   - Val/Loss: 验证损失
   - Val/Accuracy: 验证准确率
   - Val/mIoU: 平均交并比
   - Val/ClassAccuracy: 类别平均准确率
   - 每个类别的IoU
   - 可视化预测结果
"""
