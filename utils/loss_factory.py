import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from models.evidential_loss import create_evidential_loss

"""
# 示例1: 标准交叉熵损失
training:
  # ... 其他训练参数 ...
  loss:
    type: cross_entropy  # 或使用 nll
    ignore_index: 255
    class_weights: [1.0, 2.0]  # 可选: 对类别进行加权 (第二类权重更大)

# 示例2: Dice损失
training:
  # ... 其他训练参数 ...
  loss:
    type: dice
    ignore_index: 255

# 示例3: Tversky损失 (Dice损失的一个泛化版本)
training:
  # ... 其他训练参数 ...
  loss:
    type: tversky
    alpha: 0.3  # 控制假阳性的权重
    beta: 0.7   # 控制假阴性的权重
    ignore_index: 255

# 示例4: Focal损失
training:
  # ... 其他训练参数 ...
  loss:
    type: focal
    gamma: 2.0  # 聚焦参数，值越大对易分类样本的惩罚越小
    ignore_index: 255
    class_weights: [1.0, 1.5]  # 可选

# 示例5: 边界感知损失
training:
  # ... 其他训练参数 ...
  loss:
    type: boundary
    theta: 3.0  # 控制权重随距离衰减的速度
    ignore_index: 255

# 示例6: Hausdorff距离损失
training:
  # ... 其他训练参数 ...
  loss:
    type: hausdorff
    ignore_index: 255

# 示例7: 对比学习损失
training:
  # ... 其他训练参数 ...
  loss:
    type: contrastive
    temperature: 0.1  # 温度参数，控制相似度分布的平滑程度
    ignore_index: 255

# 示例8: 多损失函数组合 (Dice + 交叉熵)
training:
  # ... 其他训练参数 ...
  loss:
    type: combined
    ignore_index: 255
    losses:
      - name: cross_entropy
        weight: 1.0
      - name: dice
        weight: 0.5

# 示例9: 多损失函数组合 (边界感知损失 + 交叉熵)
training:
  # ... 其他训练参数 ...
  loss:
    type: combined
    ignore_index: 255
    losses:
      - name: cross_entropy
        weight: 0.7
      - name: boundary
        weight: 0.3
        theta: 3.0

# 示例10: 复杂组合 (交叉熵 + Dice + 边界感知)
training:
  # ... 其他训练参数 ...
  loss:
    type: combined
    ignore_index: 255
    losses:
      - name: cross_entropy
        weight: 0.5
      - name: dice
        weight: 0.3
      - name: boundary
        weight: 0.2
        theta: 5.0
"""


class LossFactory:
    """Factory class for creating loss functions based on configuration."""

    @staticmethod
    def create_loss(config):
        """
        Create a loss function from configuration.

        Args:
            config (dict): Loss configuration dictionary

        Returns:
            callable: Loss function that takes (outputs, labels) and returns loss tensor
        """
        loss_config = config.get("training", {}).get("loss", {})
        loss_type = loss_config.get("type", "nll").lower()

        # 检查是否是证据损失
        if loss_type in ["evidential", "dempster_shafer"]:
            return create_evidential_loss(loss_config)

        # Default parameters
        ignore_index = loss_config.get("ignore_index", 255)
        alpha = loss_config.get("alpha", 0.5)  # For combined losses
        weight = None

        # Handle class weights if specified
        if "class_weights" in loss_config:
            weight = torch.tensor(loss_config["class_weights"]).float().cuda()

        # Create loss function based on type
        if loss_type == "nll" or loss_type == "cross_entropy":
            # Standard NLL Loss (with LogSoftmax)
            return LossFactory._create_nll_loss(ignore_index, weight)

        elif loss_type == "dice":
            # Dice Loss
            return LossFactory._create_dice_loss(ignore_index)

        elif loss_type == "focal":
            # Focal Loss
            gamma = loss_config.get("gamma", 2.0)
            return LossFactory._create_focal_loss(gamma, ignore_index, weight)

        elif loss_type == "contrastive":
            # Pixel-wise Contrastive Loss
            temperature = loss_config.get("temperature", 0.1)
            return LossFactory._create_contrastive_loss(temperature, ignore_index)

        elif loss_type == "combined":
            # Combined loss (weighted sum of multiple losses)
            losses = []
            weights = []

            for loss_item in loss_config.get("losses", []):
                loss_name = loss_item["name"].lower()
                loss_weight = loss_item.get("weight", 1.0)

                # Create individual loss functions
                if loss_name == "nll" or loss_name == "cross_entropy":
                    losses.append(LossFactory._create_nll_loss(ignore_index, weight))
                elif loss_name == "dice":
                    losses.append(LossFactory._create_dice_loss(ignore_index))
                elif loss_name == "focal":
                    gamma = loss_item.get("gamma", 2.0)
                    losses.append(LossFactory._create_focal_loss(gamma, ignore_index, weight))
                elif loss_name == "boundary":
                    theta = loss_item.get("theta", 3.0)
                    losses.append(LossFactory._create_boundary_loss(theta, ignore_index))
                elif loss_name == "contrastive":
                    temperature = loss_item.get("temperature", 0.1)
                    losses.append(LossFactory._create_contrastive_loss(temperature, ignore_index))

                weights.append(loss_weight)

            # Create combined loss function
            return LossFactory._create_combined_loss(losses, weights, ignore_index)

        else:
            print(f"Warning: Unknown loss type '{loss_type}', defaulting to NLL loss")
            return LossFactory._create_nll_loss(ignore_index, weight)

    @staticmethod
    def _create_nll_loss(ignore_index=255, weight=None):
        """Create NLL loss function."""
        criterion = nn.NLLLoss(weight=weight, ignore_index=ignore_index).cuda()

        def loss_fn(outputs, labels):
            total = 0.0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)
                soft_output = F.log_softmax(output, dim=1).contiguous()
                total += criterion(soft_output, labels)
            return total

        return loss_fn

    @staticmethod
    def _create_dice_loss(ignore_index=255):
        """Create Dice loss function."""

        def loss_fn(outputs, labels):
            total = 0.0
            eps = 1e-6
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # probs for Dice on probabilities
                probs = F.softmax(output, dim=1)
                B, C, H, W = probs.shape

                # Build one-hot target efficiently
                valid_mask = (labels != ignore_index).unsqueeze(1)  # [B,1,H,W]
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, labels.clamp_min(0).unsqueeze(1), 1.0)  # ignore_index rows become garbage
                one_hot = one_hot * valid_mask  # zero-out ignored

                # Compute Dice per class then mean
                intersection = (probs * one_hot).sum(dim=(0, 2, 3))
                denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
                dice = (2.0 * intersection + eps) / (denom + eps)
                dice_loss = 1.0 - dice.mean()
                total += dice_loss

            return total

        return loss_fn

    @staticmethod
    def _create_focal_loss(gamma=2.0, ignore_index=255, weight=None):
        """Create Focal loss function."""

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Get probabilities with softmax
                probs = F.softmax(output, dim=1)

                # Create one-hot encoded target
                batch_size, num_classes = output.shape[0], output.shape[1]
                one_hot = torch.zeros_like(probs)

                # Handle ignore index
                valid_mask = labels != ignore_index

                # Fill one-hot tensor only for valid pixels
                for i in range(batch_size):
                    for c in range(num_classes):
                        one_hot[i, c, valid_mask[i]] = (labels[i, valid_mask[i]] == c).float()

                # Apply class weights if provided
                if weight is not None:
                    weight_tensor = weight.view(1, -1, 1, 1)
                    one_hot = one_hot * weight_tensor

                # Calculate focal loss: -(1-pt)^gamma * log(pt)
                pt = torch.sum(probs * one_hot, dim=1) + 1e-10
                focal_weight = (1 - pt) ** gamma

                # Cross entropy loss
                ce_loss = -torch.log(pt)

                # Apply focal weight to cross entropy
                focal_loss = focal_weight * ce_loss

                # Average over valid pixels
                focal_loss = torch.sum(focal_loss * valid_mask.float()) / (torch.sum(valid_mask.float()) + 1e-6)

                loss += focal_loss

            return loss

        return loss_fn

    @staticmethod
    def _create_contrastive_loss(temperature=0.1, ignore_index=255):
        """
        Create pixel-wise contrastive loss function.
        Pulls together features of pixels from the same class and pushes apart features from different classes.

        Args:
            temperature: Temperature parameter for scaling similarities
            ignore_index: Pixel value to ignore in ground truth
        """

        def loss_fn(outputs, labels):
            # Contrastive loss requires feature maps, not just final outputs
            # Assumes outputs include feature maps as auxiliary outputs

            # If no feature maps available, fall back to NLL loss
            if len(outputs) <= 1 or not hasattr(outputs, "features"):
                print("Warning: No feature maps available for contrastive loss. Falling back to NLL loss.")
                return LossFactory._create_nll_loss(ignore_index)(outputs, labels)

            # Get feature maps and output logits
            features = outputs.features  # Assume model outputs feature maps
            loss = 0

            for feat_map, output in zip(features, outputs):
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Ensure feature map is interpolated to the same size
                feat_map = F.interpolate(feat_map, size=labels.shape[1:], mode="bilinear", align_corners=False)

                batch_size, num_features, height, width = feat_map.shape

                # Normalize features
                feat_map = F.normalize(feat_map, p=2, dim=1)

                # Reshape feature map and labels for processing
                feat_map = feat_map.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_features)
                labels_flat = labels.view(batch_size, -1)

                # Handle ignore index
                valid_mask = labels_flat != ignore_index

                # Process each sample
                batch_loss = 0

                for b in range(batch_size):
                    # Get valid pixels
                    b_valid = valid_mask[b]
                    if torch.sum(b_valid) == 0:
                        continue

                    # Get features and labels for valid pixels
                    b_feat = feat_map[b, b_valid]
                    b_labels = labels_flat[b, b_valid]

                    # Compute similarity matrix
                    similarity = torch.matmul(b_feat, b_feat.transpose(0, 1)) / temperature

                    # Create mask for positive pairs (same class)
                    pos_mask = b_labels.unsqueeze(1) == b_labels.unsqueeze(0)
                    neg_mask = ~pos_mask

                    # Remove self-similarities from positive mask
                    pos_mask.fill_diagonal_(False)

                    # InfoNCE loss
                    # For each anchor, compute loss against all positives and negatives
                    exp_sim = torch.exp(similarity)

                    # Sum of exp similarities for positives
                    pos_exp_sum = torch.sum(exp_sim * pos_mask, dim=1)

                    # Sum of all exp similarities (except self)
                    all_exp_sum = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

                    # Compute loss: -log(sum_pos / sum_all)
                    # Add small epsilon to avoid log(0)
                    loss_per_anchor = -torch.log((pos_exp_sum + 1e-10) / (all_exp_sum + 1e-10))

                    # Average loss across all valid anchors
                    b_loss = torch.mean(loss_per_anchor)
                    batch_loss += b_loss

                # Average loss across batch
                loss += batch_loss / batch_size

            return loss

        return loss_fn

    @staticmethod
    def _create_combined_loss(losses, weights, ignore_index=255):
        """Create a combined loss function from multiple loss functions."""

        def loss_fn(outputs, labels):
            total_loss = 0

            for i, (loss_fn, weight) in enumerate(zip(losses, weights)):
                loss_value = loss_fn(outputs, labels)
                total_loss += weight * loss_value

            return total_loss

        return loss_fn
