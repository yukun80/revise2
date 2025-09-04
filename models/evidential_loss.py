import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLoss(nn.Module):
    """
    证据损失函数，结合分类损失和不确定性正则化
    """

    def __init__(self, num_classes=2, uncertainty_weight=0.1, focal_weight=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_weight = uncertainty_weight
        self.focal_weight = focal_weight

        # 基础分类损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, outputs, labels):
        """
        Args:
            outputs: List of [B, K, H, W] probability distributions
            labels: [B, H, W] ground truth labels
        Returns:
            total_loss: 总损失
        """
        if isinstance(outputs, list):
            output = outputs[0]  # 使用第一个输出
        else:
            output = outputs

        # 基础交叉熵损失
        ce_loss = self.ce_loss(output, labels)

        # 计算不确定性正则化
        uncertainty_loss = self._compute_uncertainty_loss(output, labels)

        # 总损失
        total_loss = ce_loss + self.uncertainty_weight * uncertainty_loss

        return total_loss

    def _compute_uncertainty_loss(self, output, labels):
        """
        计算不确定性正则化损失
        鼓励模型在困难样本上保持适度的不确定性
        """
        # 计算预测的置信度
        probs = F.softmax(output, dim=1)
        max_probs, _ = torch.max(probs, dim=1)  # [B, H, W]

        # 计算预测熵（不确定性）
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)  # [B, H, W]

        # 创建有效掩码
        valid_mask = (labels != 255).float()

        # 计算平均不确定性
        avg_uncertainty = (entropy * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        # 不确定性正则化：鼓励适度的不确定性
        # 目标：不确定性应该在0.1-0.5之间
        target_uncertainty = 0.3
        uncertainty_loss = F.mse_loss(avg_uncertainty, torch.tensor(target_uncertainty, device=output.device))

        return uncertainty_loss


class DempsterShaferLoss(nn.Module):
    """
    基于Dempster-Shafer理论的损失函数
    """

    def __init__(self, num_classes=2, uncertainty_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_weight = uncertainty_weight

    def forward(self, mass_functions, labels):
        """
        Args:
            mass_functions: [B, K+1, H, W] 质量函数 (K个类别 + 1个Theta)
            labels: [B, H, W] ground truth labels
        Returns:
            loss: Dempster-Shafer损失
        """
        B, K_plus_1, H, W = mass_functions.shape
        K = K_plus_1 - 1

        # 创建有效掩码
        valid_mask = (labels != 255).float()

        # 计算似然度函数
        plausibility = mass_functions[:, :-1, :, :] + mass_functions[:, -1:, :, :]  # [B, K, H, W]

        # 归一化为概率
        prob_dist = plausibility / (plausibility.sum(dim=1, keepdim=True) + 1e-12)

        # 计算负对数似然损失
        log_probs = torch.log(prob_dist + 1e-12)

        # 创建one-hot标签
        labels_one_hot = F.one_hot(labels.clamp(0, K - 1), num_classes=K).permute(0, 3, 1, 2).float()

        # 计算交叉熵损失
        ce_loss = -(labels_one_hot * log_probs).sum(dim=1)  # [B, H, W]
        ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        # 计算不确定性损失
        uncertainty = mass_functions[:, -1, :, :]  # [B, H, W]
        uncertainty_loss = (uncertainty * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        # 总损失
        total_loss = ce_loss + self.uncertainty_weight * uncertainty_loss

        return total_loss


def create_evidential_loss(config):
    """
    创建证据损失函数
    """
    loss_type = config.get("type", "evidential")
    num_classes = config.get("num_classes", 2)
    uncertainty_weight = config.get("uncertainty_weight", 0.1)

    if loss_type == "evidential":
        return EvidentialLoss(num_classes=num_classes, uncertainty_weight=uncertainty_weight)
    elif loss_type == "dempster_shafer":
        return DempsterShaferLoss(num_classes=num_classes, uncertainty_weight=uncertainty_weight)
    else:
        raise ValueError(f"Unknown evidential loss type: {loss_type}")


if __name__ == "__main__":
    # 测试证据损失函数
    torch.manual_seed(0)

    B, K, H, W = 2, 2, 32, 32
    outputs = [torch.randn(B, K, H, W)]
    labels = torch.randint(0, K, (B, H, W))

    # 测试EvidentialLoss
    evidential_loss = EvidentialLoss(num_classes=K)
    loss1 = evidential_loss(outputs, labels)
    print(f"Evidential Loss: {loss1.item():.4f}")

    # 测试DempsterShaferLoss
    mass_functions = torch.rand(B, K + 1, H, W)
    mass_functions = mass_functions / mass_functions.sum(dim=1, keepdim=True)  # 归一化

    ds_loss = DempsterShaferLoss(num_classes=K)
    loss2 = ds_loss(mass_functions, labels)
    print(f"Dempster-Shafer Loss: {loss2.item():.4f}")

    print("✓ 证据损失函数测试通过!")
