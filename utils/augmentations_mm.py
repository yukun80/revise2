import torch
from torch import Tensor
import torch.nn.functional as F
import random
from typing import Union, Tuple, List, Optional


class RemoteSensingModalityAugmentation:
    """为每种遥感模态提供专门的数据增强策略"""

    def __init__(self):
        # 初始化特征增强需要的卷积核
        # Sobel算子用于边缘/梯度检测
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 多尺度卷积核组
        self.kernel_sizes = [3, 5, 7]
        self.relief_kernels = {}
        for k in self.kernel_sizes:
            kernel = torch.ones((1, 1, k, k), dtype=torch.float32) / (k * k)
            self.relief_kernels[k] = kernel

        # 针对不同模态的增强参数 - 更新为特征增强型参数
        self.modality_aug_params = {
            "rgb": {
                "color_jitter": True,  # 颜色抖动
                "color_jitter_prob": 0.5,  # 颜色抖动概率
                "brightness": 0.2,  # 亮度变化范围
                "contrast": 0.2,  # 对比度变化范围
                "saturation": 0.2,  # 饱和度变化范围
            },
            "dem": {
                "terrain_enhance_prob": 0.6,  # 地形特征增强概率
                "relief_factor": 0.3,  # 局部起伏增强因子
                "multi_scale_prob": 0.4,  # 多尺度特征增强概率
                "slope_enhance_prob": 0.5,  # 坡度增强概率
                "scale_prob": 0.3,  # 高程轻微缩放概率
                "scale_range": (0.95, 1.05),  # 高程缩放范围，较小范围避免过度变形
            },
            "insar_vel": {
                "gradient_enhance_prob": 0.6,  # 形变梯度增强概率
                "gradient_factor": 0.25,  # 梯度增强因子
                "scale_prob": 0.4,  # 形变值轻微缩放概率
                "scale_range": (0.9, 1.1),  # 形变幅值缩放范围
                "edge_enhance_prob": 0.5,  # 形变边界增强概率
                "edge_factor": 0.3,  # 边界增强因子
            },
        }

        # 通用空间变换参数
        self.spatial_aug_params = {
            "rotate_prob": 0.3,  # 旋转概率
            "rotate_limit": 10,  # 旋转角度范围
            "flip_prob": 0.5,  # 翻转概率
            "scale_prob": 0.3,  # 缩放概率
            "scale_range": (0.9, 1.1),  # 缩放范围
        }

    def __call__(self, sample: dict) -> dict:
        # 1. 应用空间变换 (对所有模态一致)
        sample = self._apply_spatial_transforms(sample)

        # 2. 应用模态特定增强
        for modality, tensor in sample.items():
            if modality == "label" or modality == "metadata" or modality == "file_id":
                continue

            if modality in self.modality_aug_params:
                params = self.modality_aug_params[modality]

                # 根据模态类型应用不同增强
                if modality == "rgb" and params.get("color_jitter", False):
                    sample[modality] = self._apply_color_jitter(tensor, params)
                elif modality == "dem":
                    sample[modality] = self._apply_dem_feature_enhancement(tensor, params)
                elif modality == "insar_vel":
                    sample[modality] = self._apply_insar_feature_enhancement(tensor, params)

        return sample

    def _apply_spatial_transforms(self, sample):
        # 实现空间变换，保持一致性
        # 水平翻转
        if random.random() < self.spatial_aug_params.get("flip_prob", 0.5):
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    sample[key] = torch.flip(sample[key], [-1])

        # 旋转变换 - 保持小角度旋转以避免过度变形
        if random.random() < self.spatial_aug_params.get("rotate_prob", 0.3):
            angle = random.uniform(
                -self.spatial_aug_params.get("rotate_limit", 10), self.spatial_aug_params.get("rotate_limit", 10)
            )
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

        return sample

    def _apply_color_jitter(self, tensor, params):
        # 保留原有的RGB颜色抖动
        if random.random() < params.get("color_jitter_prob", 0.5):
            if random.random() < 0.5:
                factor = random.uniform(1 - params.get("brightness", 0.2), 1 + params.get("brightness", 0.2))
                tensor = tensor * factor

            if random.random() < 0.5:
                factor = random.uniform(1 - params.get("contrast", 0.2), 1 + params.get("contrast", 0.2))
                tensor = (tensor - 0.5) * factor + 0.5

            if random.random() < 0.5 and tensor.shape[0] == 3:
                factor = random.uniform(1 - params.get("saturation", 0.2), 1 + params.get("saturation", 0.2))
                # 计算灰度图
                gray = tensor.mean(dim=0, keepdim=True).expand_as(tensor)
                # 调整饱和度
                tensor = gray + factor * (tensor - gray)

            # 确保数值范围合理
            tensor = torch.clamp(tensor, 0, 1) if tensor.max() <= 1 else tensor

        return tensor

    def _apply_dem_feature_enhancement(self, tensor, params):
        """DEM数据特征增强，突出地形特征"""
        # 获取输入的通道数
        num_channels = tensor.shape[0]

        # 多尺度地形增强
        if random.random() < params.get("multi_scale_prob", 0.4):
            # 随机选择一个核大小
            kernel_size = random.choice(self.kernel_sizes)
            kernel = self.relief_kernels[kernel_size]

            # 处理多通道输入，逐通道应用卷积
            smoothed = torch.zeros_like(tensor)
            for c in range(num_channels):
                channel = tensor[c : c + 1].unsqueeze(0)  # [1, 1, H, W]
                # 使用2D卷积进行平滑
                smoothed[c : c + 1] = F.conv2d(channel, kernel, padding=kernel_size // 2).squeeze(0)

            # 提取局部地形起伏
            local_relief = tensor - smoothed

            # 增强局部起伏
            enhance_factor = random.uniform(0.2, params.get("relief_factor", 0.3))
            tensor = tensor + local_relief * enhance_factor

            return tensor

        # 坡度增强
        if random.random() < params.get("slope_enhance_prob", 0.5):
            # 逐通道使用Sobel算子计算坡度
            grad_x = torch.zeros_like(tensor)
            grad_y = torch.zeros_like(tensor)

            for c in range(num_channels):
                channel = tensor[c : c + 1].unsqueeze(0)  # [1, 1, H, W]
                grad_x[c : c + 1] = F.conv2d(channel, self.sobel_kernel_x, padding=1).squeeze(0)
                grad_y[c : c + 1] = F.conv2d(channel, self.sobel_kernel_y, padding=1).squeeze(0)

            # 计算坡度
            slope = torch.sqrt(grad_x**2 + grad_y**2)

            # 坡度增强
            enhance_factor = random.uniform(0.1, 0.2)  # 小系数避免过度增强
            tensor = tensor + slope * enhance_factor

            return tensor

        # 高程轻微缩放
        if random.random() < params.get("scale_prob", 0.3):
            scale_range = params.get("scale_range", (0.95, 1.05))
            scale_factor = random.uniform(*scale_range)
            # 保持平均高程不变，只调整局部变化
            mean_height = torch.mean(tensor)
            tensor = (tensor - mean_height) * scale_factor + mean_height

            return tensor

        return tensor

    def _apply_insar_feature_enhancement(self, tensor, params):
        """InSAR形变数据特征增强，突出形变特征"""
        # 获取输入的通道数
        num_channels = tensor.shape[0]

        # 形变梯度增强
        if random.random() < params.get("gradient_enhance_prob", 0.6):
            # 逐通道计算形变梯度
            grad_x = torch.zeros_like(tensor)
            grad_y = torch.zeros_like(tensor)

            for c in range(num_channels):
                channel = tensor[c : c + 1].unsqueeze(0)  # [1, 1, H, W]
                grad_x[c : c + 1] = F.conv2d(channel, self.sobel_kernel_x, padding=1).squeeze(0)
                grad_y[c : c + 1] = F.conv2d(channel, self.sobel_kernel_y, padding=1).squeeze(0)

            # 计算梯度幅值
            gradient = torch.sqrt(grad_x**2 + grad_y**2)

            # 用梯度增强原始形变场
            enhance_factor = random.uniform(0.1, params.get("gradient_factor", 0.25))
            tensor = tensor + gradient * enhance_factor

            return tensor

        # 形变边界增强
        if random.random() < params.get("edge_enhance_prob", 0.5):
            # 使用Laplacian算子增强边界
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

            # 逐通道应用Laplacian算子提取边界
            edges = torch.zeros_like(tensor)
            for c in range(num_channels):
                channel = tensor[c : c + 1].unsqueeze(0)  # [1, 1, H, W]
                edges[c : c + 1] = F.conv2d(channel, laplacian_kernel, padding=1).squeeze(0)

            # 边界增强
            enhance_factor = random.uniform(0.1, params.get("edge_factor", 0.3))
            tensor = tensor - edges * enhance_factor  # 减去因为Laplacian算子的值方向

            return tensor

        # 形变值轻微缩放
        if random.random() < params.get("scale_prob", 0.4):
            scale_range = params.get("scale_range", (0.9, 1.1))
            scale_factor = random.uniform(*scale_range)
            # 仅对非零形变区域进行缩放，保持整体形变模式
            mask = (tensor != 0).float()
            mean_val = torch.sum(tensor * mask) / (torch.sum(mask) + 1e-8)
            tensor = (tensor - mean_val * mask) * scale_factor + mean_val * mask

            return tensor

        return tensor


class ModalitySpecificNormalize:
    """为每种遥感模态提供专门的归一化策略"""

    def __init__(self, config=None):
        self.config = config
        self.modalities_to_normalize = ["dem", "insar_vel", "insar_phase", "rgb"]

        # 针对每种模态的特殊处理参数
        self.modality_params = {
            "insar_vel": {
                "clip_percentile": 99.0,  # 剪切异常值的百分比
                "use_log_scale": False,  # 是否使用对数尺度
                "balance_factor": 1.5,  # 平衡因子，提高此模态的影响
            },
            "dem": {
                "clip_percentile": 99.5,
                "use_log_scale": True,  # DEM数据通常使用对数处理效果更好
                "balance_factor": 1.2,
            },
            "rgb": {"clip_percentile": 99.0, "use_log_scale": False, "balance_factor": 1.0},  # RGB作为基准
        }

    def __call__(self, sample: dict) -> dict:
        # 对需要归一化的模态进行处理
        for modality in sample.keys():
            if modality in self.modalities_to_normalize and modality in self.config:
                # 获取模态特定参数
                params = self.modality_params.get(
                    modality, {"clip_percentile": 99.0, "use_log_scale": False, "balance_factor": 1.0}
                )

                # 应用模态特定的预处理
                tensor = sample[modality]

                # 1. 异常值处理 - 使用百分位数裁剪而非固定阈值
                if "clip_percentile" in params:
                    clip_val = torch.quantile(tensor.reshape(-1), params["clip_percentile"] / 100)
                    tensor = torch.clamp(tensor, max=clip_val)

                # 2. 对数变换处理 - 适用于DEM等数据
                if params.get("use_log_scale", False) and torch.min(tensor) >= 0:
                    tensor = torch.log1p(tensor)  # log(1+x)避免零值问题

                # 3. 获取统计参数
                num_channels = tensor.shape[0]
                has_per_band_stats = any(f"band{i+1}" in self.config[modality] for i in range(num_channels))

                # 4. 应用归一化
                if has_per_band_stats:
                    # 多波段数据，逐通道归一化
                    for channel_idx in range(num_channels):
                        band_key = f"band{channel_idx+1}"
                        if band_key in self.config[modality]:
                            band_stats = self.config[modality][band_key]

                            # 根据模态选择归一化策略
                            if modality == "rgb":
                                # RGB标准归一化
                                mean = band_stats["mean"]
                                std = band_stats["std"]
                                tensor[channel_idx] = (tensor[channel_idx] - mean) / (std + 1e-8)
                            elif modality == "insar_vel":
                                # InSAR数据特殊处理
                                min_val = band_stats["min"]
                                max_val = band_stats["max"]
                                # 扩展动态范围以增强对比度
                                tensor[channel_idx] = (tensor[channel_idx] - min_val) / (
                                    max_val - min_val + 1e-8
                                ) * 2 - 1
                            else:
                                # 其他模态使用标准min-max归一化后再标准化
                                min_val = band_stats["min"]
                                max_val = band_stats["max"]
                                mean = band_stats["mean"]
                                std = band_stats["std"]

                                tensor[channel_idx] = (tensor[channel_idx] - min_val) / (max_val - min_val + 1e-8)
                                tensor[channel_idx] = (
                                    tensor[channel_idx] - (mean - min_val) / (max_val - min_val + 1e-8)
                                ) / (std / (max_val - min_val + 1e-8) + 1e-8)
                else:
                    # 单一统计参数
                    if modality == "rgb":
                        mean = self.config[modality]["mean"]
                        std = self.config[modality]["std"]
                        tensor = (tensor - mean) / (std + 1e-8)
                    else:
                        min_val = self.config[modality]["min"]
                        max_val = self.config[modality]["max"]
                        mean = self.config[modality]["mean"]
                        std = self.config[modality]["std"]

                        tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
                        tensor = (tensor - (mean - min_val) / (max_val - min_val + 1e-8)) / (
                            std / (max_val - min_val + 1e-8) + 1e-8
                        )

                # 5. 应用平衡因子
                tensor = tensor * params.get("balance_factor", 1.0)

                # 更新样本
                sample[modality] = tensor

        return sample


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        # 验证数据格式
        assert isinstance(sample, dict), "Sample must be a dictionary"
        assert "label" in sample, "Sample must contain 'label' key"

        # 获取所有模态数据的shape
        modality_shapes = {k: v.shape for k, v in sample.items() if k not in ["metadata", "label", "file_id"]}

        # 验证所有模态数据的空间维度一致
        first_shape = next(iter(modality_shapes.values()))[1:]  # 空间维度
        for modality, shape in modality_shapes.items():
            assert shape[1:] == first_shape, f"Spatial dimensions mismatch for {modality}: {shape[1:]} vs {first_shape}"

        # 应用变换
        for transform in self.transforms:
            sample = transform(sample)

        return sample


def get_train_augmentation(config=None):
    """获取训练时的数据增强，支持选择增强策略"""
    # 默认使用遥感特定增强
    aug_strategy = "remote_sensing"

    # 从配置中读取增强策略
    if config and "augmentation" in config and "strategy" in config["augmentation"]:
        aug_strategy = config["augmentation"]["strategy"]

    if aug_strategy.lower() == "traditional":
        # 动态导入传统增强模块，避免循环导入
        from utils.augmentations_traditional import get_traditional_train_augmentation

        return get_traditional_train_augmentation(config)
    else:
        # 默认使用遥感特定增强
        return Compose(
            [
                RemoteSensingModalityAugmentation(),
                ModalitySpecificNormalize(config),
            ]
        )


def get_val_augmentation(config=None):
    """获取验证时的数据增强，支持选择增强策略"""
    # 默认使用遥感特定增强
    aug_strategy = "remote_sensing"

    # 从配置中读取增强策略
    if config and "augmentation" in config and "strategy" in config["augmentation"]:
        aug_strategy = config["augmentation"]["strategy"]

    if aug_strategy.lower() == "traditional":
        # 动态导入传统增强模块，避免循环导入
        from utils.augmentations_traditional import get_traditional_val_augmentation

        return get_traditional_val_augmentation(config)
    else:
        # 默认使用遥感特定增强
        return Compose(
            [
                ModalitySpecificNormalize(config),
            ]
        )


if __name__ == "__main__":
    h = 230
    w = 420
    sample = {}
    sample["rgb"] = torch.randn(3, h, w)
    sample["depth"] = torch.randn(3, h, w)
    sample["lidar"] = torch.randn(3, h, w)
    sample["event"] = torch.randn(3, h, w)
    sample["mask"] = torch.randn(1, h, w)
    aug = Compose(
        [
            RemoteSensingModalityAugmentation(),
        ]
    )
    sample = aug(sample)
    for k, v in sample.items():
        print(k, v.shape)
