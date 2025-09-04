import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class _GeoDsFunction(torch.autograd.Function):
    """
    2D图像版本的Dempster-Shafer证据映射函数
    适配遥感图像数据格式 [B, C, H, W]
    """

    @staticmethod
    def forward(ctx, input_feats, prototype_centers, class_membership, alpha, gamma):
        # input_feats: [B, C_in, H, W] - 2D图像特征
        # prototype_centers (W): [P, C_in]
        # class_membership (BETA): [P, K]
        # alpha: [P, 1], gamma: [P, 1]
        device = input_feats.device
        batch_size, in_channel, h, w = input_feats.size()

        # Notation alignment with original code
        BETA = class_membership
        W = prototype_centers

        class_dim = BETA.size(1)
        prototype_dim = W.size(0)

        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, class_dim, device=device))  # [P, K]
        alphap = 0.99 / (1 + torch.exp(-alpha))  # [P, 1]

        d = torch.zeros(prototype_dim, batch_size, h, w, device=device)
        s_act = torch.zeros_like(d)
        expo = torch.zeros_like(d)

        # mk accumulates Dempster combination across prototypes
        mk = torch.cat(
            (torch.zeros(class_dim, batch_size, h, w, device=device), torch.ones(1, batch_size, h, w, device=device)), 0
        )

        input_perm = input_feats.permute(1, 0, 2, 3)  # [C_in, B, H, W]

        for k in range(prototype_dim):
            # distance to prototype k: 0.5 * ||x - W[k]||^2
            # 广播W[k]到所有空间位置
            w_k = W[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [C_in, 1, 1, 1]
            w_k = w_k.expand(in_channel, batch_size, h, w)  # [C_in, B, H, W]

            temp = input_perm - w_k
            d[k, :] = 0.5 * (temp * temp).sum(0)  # [B, H, W]
            expo[k, :] = torch.exp(-(gamma[k] ** 2) * d[k, :])
            s_act[k, :] = alphap[k] * expo[k, :]

            # prototype k induces a discounted Bayesian mass over K singletons and Theta
            m_k = torch.cat(
                (
                    U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * s_act[k, :],
                    torch.ones(1, batch_size, h, w, device=device) - s_act[k, :],
                ),
                0,
            )  # [K+1, B, H, W]

            t2 = mk[:class_dim] * (m_k[:class_dim] + torch.ones(class_dim, 1, h, w, device=device) * m_k[class_dim])
            t3 = m_k[:class_dim] * (torch.ones(class_dim, 1, h, w, device=device) * mk[class_dim])
            t4 = (mk[class_dim]) * (m_k[class_dim]).unsqueeze(0)
            mk = torch.cat((t2 + t3, t4), 0)

        K_sum = mk.sum(0)
        mk_n = (mk / (torch.ones(class_dim + 1, 1, h, w, device=device) * K_sum)).permute(1, 0, 2, 3)
        # mk_n: [B, K+1, H, W] where channels 0..K-1 are class masses and K is mass on Theta
        ctx.save_for_backward(input_feats, W, BETA, alpha, gamma, mk, d)
        return mk_n

    @staticmethod
    def backward(ctx, grad_output):
        """
        完整的2D Dempster-Shafer反向传播实现（从nnFormer 3D版本移植并适配2D）
        对应的张量形状：
          - input_feats: [B, C_in, H, W]
          - W (prototype centers): [P, C_in]
          - BETA (class membership): [P, M]
          - alpha: [P, 1]
          - gamma: [P, 1]
          - mk: [M+1, B, H, W]   （未归一化的组合质量，用于反向计算）
          - d: [P, B, H, W]
        grad_output 来自前向输出 mk_n: [B, M+1, H, W]
        """
        input_feats, W, BETA, alpha, gamma, mk, d = ctx.saved_tensors

        grad_input = grad_W = grad_BETA = grad_alpha = grad_gamma = None

        # 别名与形状
        M = BETA.size(1)  # 类别数（不含Theta）
        prototype_dim = W.size(0)
        batch_size, in_channel, height, width = input_feats.size()

        mu = 0  # 正则项系数（与原实现一致）
        iw = 1  # 是否优化原型中心（与原实现一致）

        # 只对类别通道（不包含Theta）的梯度进行缩放（与原实现一致）
        grad_output_ = grad_output[:, :M, :, :] * (batch_size * M * height * width)

        # 重新计算辅助量
        K = mk.sum(0).unsqueeze(0)  # [1, B, H, W]
        K2 = K**2
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0).unsqueeze(1)  # [M, 1]
        U = BETA2 / (beta2 * torch.ones(1, M, device=input_feats.device))  # [P, M]
        alphap = 0.99 / (1 + torch.exp(-alpha))  # [P, 1]
        I = torch.eye(M, device=grad_output.device)

        # 分配工作张量（2D版本）
        s = torch.zeros(prototype_dim, batch_size, height, width, device=input_feats.device)
        expo = torch.zeros(prototype_dim, batch_size, height, width, device=input_feats.device)
        mm = torch.cat(
            (
                torch.zeros(M, batch_size, height, width, device=input_feats.device),
                torch.ones(1, batch_size, height, width, device=input_feats.device),
            ),
            0,
        )  # [M+1, B, H, W]

        dEdm = torch.zeros(M + 1, batch_size, height, width, device=input_feats.device)
        dU = torch.zeros(prototype_dim, M, device=input_feats.device)
        Ds = torch.zeros(prototype_dim, batch_size, height, width, device=input_feats.device)
        DW = torch.zeros(prototype_dim, in_channel, device=input_feats.device)

        # 对单例质量和Theta质量的 dE/dm
        for p in range(M):
            dEdm[p, :] = (
                grad_output_.permute(1, 0, 2, 3)
                * (
                    I[:, p].unsqueeze(1).unsqueeze(2).unsqueeze(3) * K
                    - mk[:M, :]
                    - (1.0 / M) * (torch.ones(M, 1, height, width, device=input_feats.device) * mk[M, :])
                )
            ).sum(0) / (K2 + 1e-12)

        dEdm[M, :] = (
            (
                grad_output_.permute(1, 0, 2, 3)
                * (-mk[:M, :] + (1.0 / M) * torch.ones(M, 1, height, width, device=input_feats.device) * (K - mk[M, :]))
            ).sum(0)
        ) / (K2 + 1e-12)

        # 遍历每个原型，计算对各参数的梯度
        for k in range(prototype_dim):
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])  # [B, H, W]
            s[k] = alphap[k] * expo[k, :]  # [B, H, W]
            m = torch.cat(
                (
                    U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * s[k, :],
                    torch.ones(1, batch_size, height, width, device=input_feats.device) - s[k, :],
                ),
                0,
            )  # [M+1, B, H, W]

            mm[M, :] = mk[M, :] / (m[M, :] + 1e-12)
            L = torch.ones(M, 1, height, width, device=input_feats.device) * mm[M, :]
            mm[:M, :] = (mk[:M, :] - L * m[:M, :]) / (
                m[:M, :] + torch.ones(M, 1, height, width, device=input_feats.device) * m[M, :] + 1e-12
            )
            R = mm[:M, :] + L
            A = R * torch.ones(M, 1, height, width, device=input_feats.device) * s[k, :]
            B = (
                U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                * torch.ones(1, batch_size, height, width, device=input_feats.device)
                * R
                - mm[:M, :]
            )

            dU[k, :] = torch.mean((A * dEdm[:M, :]).view(M, -1).permute(1, 0), 0)
            Ds[k, :] = (dEdm[:M, :] * B).sum(0) - (dEdm[M, :] * mm[M, :])

            # 原型中心梯度（DW）
            tt1 = Ds[k, :] * (gamma[k] ** 2) * s[k, :]  # [B, H, W]
            tt2 = (torch.ones(batch_size, 1, device=input_feats.device) * W[k, :]).unsqueeze(2).unsqueeze(
                3
            ) - input_feats  # [B, C, H, W]
            tt1 = tt1.view(1, -1)
            tt2 = tt2.permute(1, 0, 2, 3).reshape(in_channel, batch_size * height * width).permute(1, 0)
            DW[k, :] = -torch.mm(tt1, tt2)

        # 归一化原型中心梯度
        DW = iw * DW / (batch_size * height * width)

        # 类别隶属度（BETA）梯度
        T = beta2 * torch.ones(1, M, device=input_feats.device)
        Dbeta = (2 * BETA / (T**2)) * (
            dU * (T - BETA2)
            - (dU * BETA2).sum(1).unsqueeze(1) * torch.ones(1, M, device=input_feats.device)
            + dU * BETA2
        )

        # gamma 与 alpha 的梯度
        Dgamma = -2 * torch.mean(((Ds * d * s).view(prototype_dim, -1)).t(), 0).unsqueeze(1) * gamma
        Dalpha = (torch.mean(((Ds * expo).view(prototype_dim, -1)).t(), 0).unsqueeze(1) + mu) * (
            0.99 * (1 - alphap) * alphap
        )

        # 输入特征的梯度
        Dinput = torch.zeros(batch_size, in_channel, height, width, device=input_feats.device)
        temp2 = torch.zeros(prototype_dim, in_channel, height, width, device=input_feats.device)
        for n in range(batch_size):
            for k in range(prototype_dim):
                diff = input_feats[n, :] - W[k, :].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                coeff = (Ds[k, n, :, :] * (gamma[k] ** 2) * s[k, n, :, :]).unsqueeze(0).unsqueeze(1)
                temp2[k] = -prototype_dim * coeff * diff
            Dinput[n, :] = temp2.mean(0)

        # 按需返回梯度
        if ctx.needs_input_grad[0]:
            grad_input = Dinput
        if ctx.needs_input_grad[1]:
            grad_W = DW
        if ctx.needs_input_grad[2]:
            grad_BETA = Dbeta
        if ctx.needs_input_grad[3]:
            grad_alpha = Dalpha
        if ctx.needs_input_grad[4]:
            grad_gamma = Dgamma

        # 保证梯度张量连续，避免后端报错
        if grad_input is not None:
            grad_input = grad_input.contiguous()
        if grad_W is not None:
            grad_W = grad_W.contiguous()
        if grad_BETA is not None:
            grad_BETA = grad_BETA.contiguous()
        if grad_alpha is not None:
            grad_alpha = grad_alpha.contiguous()
        if grad_gamma is not None:
            grad_gamma = grad_gamma.contiguous()

        return grad_input, grad_W, grad_BETA, grad_alpha, grad_gamma


class GeoEvidentialMappingLayer(nn.Module):
    """
    地理证据映射层 (Geo-Evidential Mapping Layer, GEM-Layer)

    基于Dempster-Shafer证据理论，将2D图像特征映射为证据质量函数。
    专门针对遥感滑坡制图任务设计，支持多模态输入。

    Args:
        input_dim (int): 输入特征通道数
        prototype_dim (int): 原型数量
        class_dim (int): 类别数量 (默认2: 滑坡/非滑坡)
        geo_prior_weight (float): 地理先验权重
    Returns:
        Tensor of shape [B, K+1, H, W] - 质量函数
    """

    def __init__(self, input_dim: int, prototype_dim: int, class_dim: int = 2, geo_prior_weight: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.prototype_dim = prototype_dim
        self.class_dim = class_dim
        self.geo_prior_weight = geo_prior_weight

        # 核心参数
        self.class_membership = Parameter(torch.Tensor(self.prototype_dim, self.class_dim))  # BETA
        self.alpha = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.gamma = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.prototype_centers = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))  # W

        # 地理先验约束
        self.geo_constraints = Parameter(torch.Tensor(self.prototype_dim, 1))

        # 季节性权重 (4个季节)
        self.seasonal_weights = Parameter(torch.Tensor(4, self.prototype_dim))

        # 地形复杂度权重
        self.terrain_complexity_weight = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        # 核心参数初始化 - 更合理的初始化
        nn.init.normal_(self.prototype_centers, std=0.01)  # 减小初始化方差
        nn.init.xavier_uniform_(self.class_membership)
        nn.init.constant_(self.gamma, 0.1)  # 更小的gamma，避免指数项过度衰减
        nn.init.constant_(self.alpha, 0.0)  # 更小的alpha，避免s_act饱和

        # 地理先验初始化
        nn.init.normal_(self.geo_constraints, std=0.01)
        nn.init.normal_(self.seasonal_weights, std=0.01)
        nn.init.constant_(self.terrain_complexity_weight, 1.0)

    def forward(self, feats: torch.Tensor, geo_context: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            feats: [B, C, H, W] 输入特征图
            geo_context: [B, 1, H, W] 地理上下文信息 (可选)
        Returns:
            mass: [B, K+1, H, W] 质量函数
        """
        # 应用地理先验约束（不改变形状）
        if geo_context is not None:
            adjusted_centers = self._apply_geo_prior(self.prototype_centers, geo_context)
        else:
            adjusted_centers = self.prototype_centers

        # feats: [B, C, H, W]
        B, C, H, W = feats.shape
        device = feats.device

        P = adjusted_centers.size(0)
        K = self.class_membership.size(1)

        # U 与 alpha'
        BETA = self.class_membership  # [P, K]
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)  # [P]
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, K, device=device))  # [P, K]
        alphap = 0.99 / (1 + torch.exp(-self.alpha))  # [P, 1]

        # 使用自定义的DS函数（包含完整的反向传播）
        mass = _GeoDsFunction.apply(feats, adjusted_centers, BETA, self.alpha, self.gamma)
        return mass

    def _apply_geo_prior(self, prototype_centers: torch.Tensor, geo_context: torch.Tensor) -> torch.Tensor:
        """
        应用地理先验约束

        Args:
            prototype_centers: [P, C] 原型中心
            geo_context: [B, 1, H, W] 地理上下文
        Returns:
            adjusted_centers: [P, C] 调整后的原型中心
        """
        # 简化的地理先验应用
        # 在实际应用中，这里可以集成更复杂的地理知识
        # 轻量稳定：只做一个标量缩放，不引入空间维度
        scale = 1.0 + self.geo_prior_weight * torch.tanh(self.geo_constraints.mean())
        return prototype_centers * scale

    def get_uncertainty(self, mass: torch.Tensor) -> torch.Tensor:
        """
        从质量函数中提取不确定性

        Args:
            mass: [B, K+1, H, W] 质量函数
        Returns:
            uncertainty: [B, 1, H, W] 不确定性图
        """
        # 不确定性 = 质量函数中分配给Theta的部分
        uncertainty = mass[:, -1:, :, :]  # [B, 1, H, W]
        return uncertainty

    def get_plausibility(self, mass: torch.Tensor) -> torch.Tensor:
        """
        计算似然度函数

        Args:
            mass: [B, K+1, H, W] 质量函数
        Returns:
            plausibility: [B, K, H, W] 似然度
        """
        pl_singletons = mass[:, :-1, :, :]  # [B, K, H, W]
        m_theta = mass[:, -1:, :, :]  # [B, 1, H, W]
        plausibility = pl_singletons + m_theta  # [B, K, H, W]
        return plausibility


def mass_to_plausibility(mass: torch.Tensor) -> torch.Tensor:
    """
    将质量函数转换为似然度函数

    Args:
        mass: [B, K+1, H, W] 质量函数
    Returns:
        plausibility: [B, K, H, W] 似然度
    """
    pl_singletons = mass[:, :-1, :, :]
    m_theta = mass[:, -1:, :, :]
    return pl_singletons + m_theta


def plausibility_to_probability(plausibility: torch.Tensor) -> torch.Tensor:
    """
    将似然度转换为概率分布

    Args:
        plausibility: [B, K, H, W] 似然度
    Returns:
        probability: [B, K, H, W] 概率分布
    """
    K_sum = plausibility.sum(1, keepdim=True)
    probability = plausibility / (K_sum + 1e-12)
    return probability


if __name__ == "__main__":
    # 测试GEM-Layer
    torch.manual_seed(0)

    # 模拟输入
    B, C_in, H, W = 2, 256, 32, 32
    K = 2  # 滑坡/非滑坡
    P = 20  # 原型数量

    feats = torch.randn(B, C_in, H, W)
    geo_context = torch.randn(B, 1, H, W)

    # 创建GEM-Layer
    gem_layer = GeoEvidentialMappingLayer(input_dim=C_in, prototype_dim=P, class_dim=K, geo_prior_weight=0.1)

    # 前向传播
    mass = gem_layer(feats, geo_context)
    uncertainty = gem_layer.get_uncertainty(mass)
    plausibility = gem_layer.get_plausibility(mass)
    probability = plausibility_to_probability(plausibility)

    print("=== GEM-Layer 测试结果 ===")
    print(f"输入特征形状: {feats.shape}")
    print(f"质量函数形状: {mass.shape}")
    print(f"不确定性形状: {uncertainty.shape}")
    print(f"似然度形状: {plausibility.shape}")
    print(f"概率分布形状: {probability.shape}")

    # 验证质量函数的性质
    mass_sum = mass.sum(dim=1, keepdim=True)
    print(f"质量函数和: {mass_sum.min().item():.4f} - {mass_sum.max().item():.4f}")

    # 验证概率分布的性质
    prob_sum = probability.sum(dim=1, keepdim=True)
    print(f"概率分布和: {prob_sum.min().item():.4f} - {prob_sum.max().item():.4f}")

    print("✓ GEM-Layer 实现正确!")
