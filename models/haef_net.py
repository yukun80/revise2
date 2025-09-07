import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer import WeTr
from .geo_evidential_mapping import GeoEvidentialMappingLayer
from .modality_reliability_gating import ModalityReliabilityGating


class HAEFNet(nn.Module):
    """
    Hierarchical Attentive & Evidential Fusion Network (HAEF-Net)
    层次化注意与证据融合网络

    集成HARMF的注意力机制和证据理论的决策级融合
    """

    def __init__(
        self,
        backbone="swin_tiny",
        num_classes=2,
        n_heads=8,
        dpr=0.1,
        drop_rate=0.0,
        num_parallel=3,  # RGB, InSAR, DEM
        fusion_params=None,
        # GEM-Layer参数
        gem_prototype_dim=20,
        gem_geo_prior_weight=0.1,
        # 是否使用证据融合
        use_evidential_fusion=True,
        use_mrg=False,
        # 概率融合策略：product | mean | weighted_product
        prob_fusion: str = "product",
        # 新增：是否在GEM前保留PCA跨模态交互更新
        keep_pca_before_gem: bool = True,
        # 新增：单尺度聚合通道
        aggregation_channels: int = 256,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_parallel = num_parallel
        self.use_evidential_fusion = use_evidential_fusion
        self.use_mrg = use_mrg
        self.prob_fusion = prob_fusion
        self.keep_pca_before_gem = keep_pca_before_gem
        self.aggregation_channels = aggregation_channels

        print("-----------------HAEF-Net Params--------------------------------------")
        print("backbone:", backbone)
        print("num_classes:", num_classes)
        print("num_modalities:", num_parallel)
        print("use_evidential_fusion:", use_evidential_fusion)
        print("gem_prototype_dim:", gem_prototype_dim)
        print("prob_fusion:", self.prob_fusion)
        print("--------------------------------------------------------------")

        # 1. HARMF编码器 (保持原有功能)
        self.harmf_encoder = WeTr(
            backbone=backbone,
            num_classes=num_classes,
            n_heads=n_heads,
            dpr=dpr,
            drop_rate=drop_rate,
            num_parallel=num_parallel,
            fusion_params=fusion_params,
        )

        # 获取编码器输出特征维度
        if "swin" in backbone:
            if backbone == "swin_tiny":
                self.feature_dims = [96, 192, 384, 768]
            elif backbone == "swin_small":
                self.feature_dims = [96, 192, 384, 768]
            elif backbone == "swin_large":
                self.feature_dims = [192, 384, 768, 1536]
            else:
                self.feature_dims = [192, 384, 768, 1536]
        else:
            # 对于MiT backbone，使用默认维度
            self.feature_dims = [128, 256, 512, 1024]

        # 2. 单尺度聚合 + 单次GEM（方案B）
        if self.use_evidential_fusion:
            # 顶向下聚合器：逐stage 1x1降维到统一通道，并上采样到stage1尺寸后融合
            self.agg_proj = nn.ModuleList(
                [nn.Conv2d(dim, self.aggregation_channels, kernel_size=1) for dim in self.feature_dims]
            )
            # 融合层（concat后1x1更稳；如需极致速度可改为逐stage求和）
            self.agg_fuse = nn.Conv2d(self.aggregation_channels * len(self.feature_dims), self.aggregation_channels, 1)

            # 单次 GEM（对每个模态各做一次）
            self.gem_single = GeoEvidentialMappingLayer(
                input_dim=self.aggregation_channels,
                prototype_dim=gem_prototype_dim,
                class_dim=num_classes,
                geo_prior_weight=gem_geo_prior_weight,
            )

            if self.use_mrg:
                self.mrg = ModalityReliabilityGating(num_classes=self.num_classes, num_modalities=self.num_parallel)

    def get_param_groups(self):
        """获取参数组，用于优化器设置"""
        param_groups = [[], [], []]  # encoder, norm, decoder

        # HARMF编码器参数
        for name, param in list(self.harmf_encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # 聚合与GEM参数 (使用更高学习率)
        if self.use_evidential_fusion:
            for param in self.agg_proj.parameters():
                param_groups[2].append(param)
            for param in self.agg_fuse.parameters():
                param_groups[2].append(param)
            for param in self.gem_single.parameters():
                param_groups[2].append(param)
            # MRG 参数
            if self.use_mrg:
                for param in self.mrg.parameters():
                    param_groups[2].append(param)

        return param_groups

    def forward(self, x):
        """
        前向传播

        Args:
            x: List of modality tensors [RGB, InSAR, DEM], each [B, C, H, W]
        Returns:
            outputs: List of output tensors
            aux_loss: Auxiliary loss (for compatibility with HARMF)
        """
        # 记录原始输入尺寸
        original_shape = x[0].shape[2:]

        # 若未启用证据融合，走原HARMF路径
        if not self.use_evidential_fusion:
            return self.harmf_encoder(x)

        # 仅获取编码器多模态多尺度特征（不走WeTr.forward解码路径）
        modality_features = self.harmf_encoder.encoder(x)  # [[c1..c4]_m0, [c1..c4]_m1, ...]
        num_modalities = len(modality_features)
        if num_modalities == 0:
            return self.harmf_encoder(x)

        # 可选：在GEM前做一次像素级跨模态注意交互更新（与WeTr.forward同源）
        updated_per_modality = [[] for _ in range(num_modalities)]
        num_stages = len(self.feature_dims)
        for s in range(num_stages):
            stage_feats = []
            for m in range(num_modalities):
                if s < len(modality_features[m]):
                    stage_feats.append(modality_features[m][s])
            if not stage_feats:
                continue
            B, C, Hs, Ws = stage_feats[0].shape
            if self.keep_pca_before_gem:
                seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
                updated_seqs = list(seqs)
                for m in range(num_modalities):
                    nxt = (m + 1) % num_modalities
                    y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                    updated_seqs[m], updated_seqs[nxt] = y_m, y_n
                stage_feats = [
                    updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
                    for m in range(num_modalities)
                ]
            for m in range(num_modalities):
                updated_per_modality[m].append(stage_feats[m])

        # 每模态：多尺度 -> 单尺度（对齐stage1尺寸），再做一次GEM
        pl_list = []  # 存储每模态的折扣后轮廓函数 [B,K,H/4,W/4]
        theta_list = []  # 存储每模态的不确定性 [B,1,H/4,W/4]
        target_hw = None
        # 目标尺寸：以stage1为准
        for m in range(num_modalities):
            if len(updated_per_modality[m]) > 0:
                target_hw = updated_per_modality[m][0].shape[2:]
                break
        if target_hw is None:
            return self.harmf_encoder(x)

        for t in range(self.num_parallel):
            if t >= num_modalities or len(updated_per_modality[t]) == 0:
                continue
            proj_ups = []
            for s, feat in enumerate(updated_per_modality[t]):
                z = self.agg_proj[s](feat)
                if z.shape[2:] != target_hw:
                    z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
                proj_ups.append(z)
            agg_t = self.agg_fuse(torch.cat(proj_ups, dim=1))  # [B, Cagg, H/4, W/4]

            # 单次 GEM
            mass_t = self.gem_single(agg_t)  # [B, K+1, H/4, W/4]
            theta_t = self.gem_single.get_uncertainty(mass_t)  # [B,1,H/4,W/4]
            pl_t = self.gem_single.get_plausibility(mass_t)  # [B,K,H/4,W/4]

            # 情境折扣（在pl域）：pl_hat = 1 - beta + beta * pl
            if self.use_mrg:
                pl_t = self.mrg(pl_t, modality_index=t)

            pl_list.append(pl_t)
            theta_list.append(theta_t)

        # 跨模态DS等价组合（轮廓函数乘积后归一化得到概率）
        if not pl_list:
            return self.harmf_encoder(x)

        pl_fused = pl_list[0]
        for i in range(1, len(pl_list)):
            pl_fused = pl_fused * pl_list[i]
        prob = pl_fused / (pl_fused.sum(1, keepdim=True) + 1e-12)  # [B,K,H/4,W/4]

        # logits 与上采样到原始尺寸
        logits = torch.log(prob + 1e-12)
        if logits.shape[2:] != original_shape:
            logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
        outputs_all = [logits]
        return outputs_all, None

    def get_uncertainty_map(self, x):
        """
        获取不确定性图

        Args:
            x: List of modality tensors
        Returns:
            uncertainty_map: [B, 1, H, W] 不确定性图
        """
        if not self.use_evidential_fusion:
            return None

        with torch.no_grad():
            modality_features = self.harmf_encoder.encoder(x)
            num_modalities = len(modality_features)
            if num_modalities == 0:
                return None

            # 可选PCA
            updated_per_modality = [[] for _ in range(num_modalities)]
            num_stages = len(self.feature_dims)
            for s in range(num_stages):
                stage_feats = []
                for m in range(num_modalities):
                    if s < len(modality_features[m]):
                        stage_feats.append(modality_features[m][s])
                if not stage_feats:
                    continue
                B, C, Hs, Ws = stage_feats[0].shape
                if self.keep_pca_before_gem:
                    seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
                    updated_seqs = list(seqs)
                    for m in range(num_modalities):
                        nxt = (m + 1) % num_modalities
                        y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                        updated_seqs[m], updated_seqs[nxt] = y_m, y_n
                    stage_feats = [
                        updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
                        for m in range(num_modalities)
                    ]
                for m in range(num_modalities):
                    updated_per_modality[m].append(stage_feats[m])

            # 目标尺寸：stage1
            target_hw = None
            for m in range(num_modalities):
                if len(updated_per_modality[m]) > 0:
                    target_hw = updated_per_modality[m][0].shape[2:]
                    break
            if target_hw is None:
                return None

            theta_list = []
            for t in range(self.num_parallel):
                if t >= num_modalities or len(updated_per_modality[t]) == 0:
                    continue
                proj_ups = []
                for s, feat in enumerate(updated_per_modality[t]):
                    z = self.agg_proj[s](feat)
                    if z.shape[2:] != target_hw:
                        z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
                    proj_ups.append(z)
                agg_t = self.agg_fuse(torch.cat(proj_ups, dim=1))
                mass_t = self.gem_single(agg_t)
                theta_t = self.gem_single.get_uncertainty(mass_t)
                theta_list.append(theta_t)

            if not theta_list:
                return None
            U = torch.stack(theta_list, dim=0).mean(dim=0)  # [B,1,H/4,W/4]
            if U.shape[2:] != x[0].shape[2:]:
                U = F.interpolate(U, size=x[0].shape[2:], mode="bilinear", align_corners=False)
            return U

    @torch.no_grad()
    def analyze_modalities(self, x, foreground_class: int = 1, compute_loo: bool = True):
        """
        分析各模态贡献与不确定性，并可选计算留一法(LOO)融合结果（与单尺度GEM+DS决策路径对齐）。

        Returns dict with:
          - final_logits: [B,K,H,W]
          - final_prob: [B,K,H,W]
          - U: [B,1,H,W] 最终不确定性
          - C_t: [B,T,H,W] 每模态log贡献(前景类)
          - U_t: [B,T,H,W] 每模态不确定性
          - prob_loo: list len T of [B,K,H,W] (可选)
          - beta: [T,K] 可靠性参数sigmoid
        """
        eps = 1e-12
        original_shape = x[0].shape[2:]

        # 提取多模态多尺度特征
        modality_features = self.harmf_encoder.encoder(x)  # [[c1..c4]_m0, [c1..c4]_m1, ...]
        num_modalities = len(modality_features)
        if num_modalities == 0:
            final_logits, _ = self.forward(x)
            final_logits = final_logits[0]
            final_prob = F.softmax(final_logits, dim=1)
            return {
                "final_logits": final_logits,
                "final_prob": final_prob,
                "U": None,
                "C_t": None,
                "U_t": None,
                "prob_loo": None,
                "beta": None,
            }

        # 可选：PCA跨模态交互更新
        updated_per_modality = [[] for _ in range(num_modalities)]
        num_stages = len(self.feature_dims)
        for s in range(num_stages):
            stage_feats = []
            for m in range(num_modalities):
                if s < len(modality_features[m]):
                    stage_feats.append(modality_features[m][s])
            if not stage_feats:
                continue
            B, C, Hs, Ws = stage_feats[0].shape
            if self.keep_pca_before_gem:
                seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
                updated_seqs = list(seqs)
                for m in range(num_modalities):
                    nxt = (m + 1) % num_modalities
                    y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                    updated_seqs[m], updated_seqs[nxt] = y_m, y_n
                stage_feats = [
                    updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
                    for m in range(num_modalities)
                ]
            for m in range(num_modalities):
                updated_per_modality[m].append(stage_feats[m])

        # 目标尺寸：stage1分辨率
        target_hw = None
        for m in range(num_modalities):
            if len(updated_per_modality[m]) > 0:
                target_hw = updated_per_modality[m][0].shape[2:]
                break
        if target_hw is None:
            final_logits, _ = self.forward(x)
            final_logits = final_logits[0]
            final_prob = F.softmax(final_logits, dim=1)
            return {
                "final_logits": final_logits,
                "final_prob": final_prob,
                "U": None,
                "C_t": None,
                "U_t": None,
                "prob_loo": None,
                "beta": None,
            }

        # 每模态单尺度聚合 + 单次GEM + 情境折扣
        pl_list = []  # 折扣后pl，形状 [B,K,h,w]
        prob_t_list = []  # 每模态自身概率（用于贡献图）
        theta_up_list = []  # 上采样到原尺寸的不确定性
        C_list = []  # 每模态贡献图（前景log prob）

        for t in range(self.num_parallel):
            if t >= num_modalities or len(updated_per_modality[t]) == 0:
                continue
            proj_ups = []
            for s, feat in enumerate(updated_per_modality[t]):
                z = self.agg_proj[s](feat)
                if z.shape[2:] != target_hw:
                    z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
                proj_ups.append(z)
            agg_t = self.agg_fuse(torch.cat(proj_ups, dim=1))

            mass_t = self.gem_single(agg_t)
            theta_t = self.gem_single.get_uncertainty(mass_t)  # [B,1,h,w]
            pl_t = self.gem_single.get_plausibility(mass_t)  # [B,K,h,w]
            if self.use_mrg:
                pl_t = self.mrg(pl_t, modality_index=t)

            # modality probability（用于贡献图）
            prob_t = pl_t / (pl_t.sum(1, keepdim=True) + eps)
            prob_t_list.append(prob_t)
            pl_list.append(pl_t)

            # per-modality U、C_t 上采样到原尺寸
            theta_up = theta_t
            if theta_up.shape[2:] != original_shape:
                theta_up = F.interpolate(theta_up, size=original_shape, mode="bilinear", align_corners=False)
            theta_up_list.append(theta_up)

            c_map = torch.log(prob_t[:, foreground_class : foreground_class + 1, :, :] + 1e-8)  # [B,1,h,w]
            if c_map.shape[2:] != original_shape:
                c_map = F.interpolate(c_map, size=original_shape, mode="bilinear", align_corners=False)
            C_list.append(c_map.squeeze(1))  # [B,H,W]

        # DS等价融合：pl乘积 -> 概率
        pl_fused = pl_list[0]
        for i in range(1, len(pl_list)):
            pl_fused = pl_fused * pl_list[i]
        prob_small = pl_fused / (pl_fused.sum(1, keepdim=True) + eps)
        # 上采样并重归一
        prob_up = prob_small
        if prob_up.shape[2:] != original_shape:
            prob_up = F.interpolate(prob_up, size=original_shape, mode="bilinear", align_corners=False)
            prob_up = prob_up / (prob_up.sum(1, keepdim=True) + eps)
        final_prob = prob_up
        final_logits = torch.log(final_prob + eps)

        # 最终不确定性：各模态theta均值
        U = torch.stack(theta_up_list, dim=0).mean(dim=0) if theta_up_list else None  # [B,1,H,W]

        # 汇总每模态统计
        C_t = torch.stack(C_list, dim=1) if C_list else None  # [B,T,H,W]
        U_t = torch.cat(theta_up_list, dim=1) if theta_up_list else None  # [B,T,H,W]（每模态1通道拼接）

        # 留一法：对每个t，融合其它模态pl
        prob_loo = None
        if compute_loo and len(pl_list) > 1:
            prob_loo = []
            for t in range(len(pl_list)):
                # 构建除t之外的乘积
                pl_others = None
                for j, pl_j in enumerate(pl_list):
                    if j == t:
                        continue
                    pl_others = pl_j if pl_others is None else pl_others * pl_j
                prob_no_t = pl_others / (pl_others.sum(1, keepdim=True) + eps)
                if prob_no_t.shape[2:] != original_shape:
                    prob_no_t = F.interpolate(prob_no_t, size=original_shape, mode="bilinear", align_corners=False)
                    prob_no_t = prob_no_t / (prob_no_t.sum(1, keepdim=True) + eps)
                prob_loo.append(prob_no_t)

        beta = None
        if hasattr(self, "mrg") and self.use_mrg:
            beta = torch.sigmoid(self.mrg.alpha).squeeze(-1).squeeze(-1)  # [T,K]

        return {
            "final_logits": final_logits,
            "final_prob": final_prob,
            "U": U,
            "C_t": C_t,
            "U_t": U_t,
            "prob_loo": prob_loo,
            "beta": beta,
        }


if __name__ == "__main__":
    # 测试HAEF-Net
    torch.manual_seed(0)

    # 模拟输入
    B, C, H, W = 2, 3, 64, 64
    x = [torch.randn(B, C, H, W) for _ in range(3)]  # RGB, InSAR, DEM

    # 创建HAEF-Net
    model = HAEFNet(
        backbone="swin_tiny", num_classes=2, num_parallel=3, use_evidential_fusion=True, gem_prototype_dim=10
    )

    # 前向传播
    outputs, aux_loss = model(x)
    uncertainty_map = model.get_uncertainty_map(x)

    print("=== HAEF-Net 测试结果 ===")
    print(f"输入模态数量: {len(x)}")
    print(f"输入形状: {[xi.shape for xi in x]}")
    print(f"输出数量: {len(outputs)}")
    print(f"输出形状: {[out.shape for out in outputs]}")
    print(f"不确定性图形状: {uncertainty_map.shape if uncertainty_map is not None else 'None'}")
    print(f"辅助损失: {aux_loss}")

    print("✓ HAEF-Net 实现正确!")
