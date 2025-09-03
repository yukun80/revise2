import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Union, Tuple, List, Optional


class TraditionalImageAugmentation:
    """用于自然图像处理的传统数据增强方法，适用于各种模型"""

    def __init__(self, config=None):
        # 设置默认参数
        self.aug_params = {
            "brightness": 0.2,  # 亮度变化范围
            "contrast": 0.2,  # 对比度变化范围
            "saturation": 0.2,  # 饱和度变化范围
            "hflip_prob": 0.5,  # 水平翻转概率
            "vflip_prob": 0.3,  # 垂直翻转概率
            "rotate_prob": 0.3,  # 旋转概率
            "rotate_limit": 15,  # 旋转角度限制
            "scale_prob": 0.3,  # 缩放概率
            "scale_range": [0.9, 1.1],  # 缩放范围
        }

        # 如果提供了配置，则更新默认参数
        if config and "augmentation" in config and "traditional" in config["augmentation"]:
            traditional_config = config["augmentation"]["traditional"]
            for key, value in traditional_config.items():
                if key in self.aug_params:
                    self.aug_params[key] = value

    def __call__(self, sample: dict) -> dict:
        """应用传统的数据增强策略"""
        # 首先应用空间变换 (对所有模态一致)
        sample = self._apply_spatial_transforms(sample)

        # 应用颜色增强 (仅对RGB模态)
        for modality, tensor in sample.items():
            if modality == "label" or modality == "metadata" or modality == "file_id":
                continue

            if modality == "rgb":
                sample[modality] = self._apply_color_jitter(tensor)

        return sample

    def _apply_spatial_transforms(self, sample):
        """应用空间变换，保持跨模态一致性"""
        # 水平翻转
        if random.random() < self.aug_params["hflip_prob"]:
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    sample[key] = torch.flip(sample[key], [-1])

        # 垂直翻转 (对遥感可能不适用，但对一般图像可用)
        if random.random() < self.aug_params["vflip_prob"]:
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    sample[key] = torch.flip(sample[key], [-2])

        # 旋转变换
        if random.random() < self.aug_params["rotate_prob"]:
            angle = random.uniform(-self.aug_params["rotate_limit"], self.aug_params["rotate_limit"])
            # 旋转矩阵
            theta = torch.tensor(
                [
                    [
                        torch.cos(torch.tensor(angle * torch.pi / 180)),
                        -torch.sin(torch.tensor(angle * torch.pi / 180)),
                        0,
                    ],
                    [
                        torch.sin(torch.tensor(angle * torch.pi / 180)),
                        torch.cos(torch.tensor(angle * torch.pi / 180)),
                        0,
                    ],
                ],
                dtype=torch.float,
            )

            # 对所有模态进行相同的旋转
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    if key == "label":
                        # 对标签使用最近邻插值
                        mode = "nearest"
                    else:
                        # 对特征使用双线性插值
                        mode = "bilinear"

                    # 获取尺寸
                    if key == "label":
                        h, w = sample[key].shape
                        # 旋转标签（需要先扩展维度）
                        sample[key] = sample[key].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        grid = F.affine_grid(theta.unsqueeze(0), sample[key].size(), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].float(), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0).squeeze(0).long()  # 转回原始维度和类型
                    else:
                        # 获取特征尺寸
                        c, h, w = sample[key].shape
                        # 旋转特征
                        grid = F.affine_grid(theta.unsqueeze(0), torch.Size([1, c, h, w]), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].unsqueeze(0), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0)

        # 随机缩放
        if random.random() < self.aug_params["scale_prob"]:
            scale = random.uniform(self.aug_params["scale_range"][0], self.aug_params["scale_range"][1])

            for key in sample:
                if key not in ["metadata", "file_id"]:
                    if key == "label":
                        # 对标签使用最近邻插值
                        mode = "nearest"
                    else:
                        # 对特征使用双线性插值
                        mode = "bilinear"

                    # 获取尺寸
                    if key == "label":
                        h, w = sample[key].shape
                        # 缩放标签
                        sample[key] = sample[key].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                        # 创建缩放矩阵
                        theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], dtype=torch.float).unsqueeze(0)

                        grid = F.affine_grid(theta, sample[key].size(), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].float(), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0).squeeze(0).long()  # 转回原始维度和类型
                    else:
                        # 获取特征尺寸
                        c, h, w = sample[key].shape

                        # 创建缩放矩阵
                        theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], dtype=torch.float).unsqueeze(0)

                        grid = F.affine_grid(theta, torch.Size([1, c, h, w]), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].unsqueeze(0), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0)

        return sample

    def _apply_color_jitter(self, tensor):
        """应用颜色抖动（仅RGB模态）"""
        # 亮度调整
        if random.random() < 0.5:
            factor = random.uniform(1 - self.aug_params["brightness"], 1 + self.aug_params["brightness"])
            tensor = tensor * factor

        # 对比度调整
        if random.random() < 0.5:
            factor = random.uniform(1 - self.aug_params["contrast"], 1 + self.aug_params["contrast"])
            tensor = (tensor - 0.5) * factor + 0.5

        # 饱和度调整 (仅对3通道RGB)
        if random.random() < 0.5 and tensor.shape[0] == 3:
            factor = random.uniform(1 - self.aug_params["saturation"], 1 + self.aug_params["saturation"])
            # 计算灰度图
            gray = tensor.mean(dim=0, keepdim=True).expand_as(tensor)
            # 调整饱和度
            tensor = gray + factor * (tensor - gray)

        # 确保数值范围合理
        tensor = torch.clamp(tensor, 0, 1) if tensor.max() <= 1 else tensor

        return tensor


class TraditionalNormalize:
    """简化版的归一化，适用于传统图像处理模型"""

    def __init__(self, config=None):
        self.config = config
        self.modalities_to_normalize = ["dem", "insar_vel", "insar_phase", "rgb"]

        # 默认归一化参数
        self.default_norm = {
            "rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet统计量
            "dem": {"mean": 0.0, "std": 1.0},
            "insar_vel": {"mean": 0.0, "std": 1.0},
        }

    def __call__(self, sample: dict) -> dict:
        # 对需要归一化的模态进行处理
        for modality in sample.keys():
            if modality in self.modalities_to_normalize:
                if modality not in ["metadata", "file_id", "label"]:
                    tensor = sample[modality]

                    # 获取归一化参数
                    norm_params = self.default_norm.get(modality, {"mean": 0.0, "std": 1.0})

                    # 如果配置文件中有参数，优先使用配置文件的参数
                    if self.config and modality in self.config:
                        if "mean" in self.config[modality]:
                            norm_params["mean"] = self.config[modality]["mean"]
                        if "std" in self.config[modality]:
                            norm_params["std"] = self.config[modality]["std"]

                    # 应用归一化
                    if modality == "rgb" and isinstance(norm_params["mean"], list):
                        # 多通道RGB归一化
                        mean = torch.tensor(norm_params["mean"]).view(3, 1, 1)
                        std = torch.tensor(norm_params["std"]).view(3, 1, 1)
                        tensor = (tensor - mean) / std
                    else:
                        # 单通道归一化，或RGB使用相同参数
                        mean = norm_params["mean"]
                        std = norm_params["std"]
                        tensor = (tensor - mean) / std

                    sample[modality] = tensor

        return sample


# 导入原始Compose类以保持一致性
from utils.augmentations_mm import Compose


def get_traditional_train_augmentation(config=None):
    """获取传统的训练增强策略"""
    return Compose(
        [
            TraditionalImageAugmentation(config),
            TraditionalNormalize(config),
        ]
    )


def get_traditional_val_augmentation(config=None):
    """获取传统的验证增强策略"""
    return Compose(
        [
            TraditionalNormalize(config),
        ]
    )
