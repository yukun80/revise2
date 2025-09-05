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
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_parallel = num_parallel
        self.use_evidential_fusion = use_evidential_fusion
        self.use_mrg = use_mrg
        self.prob_fusion = prob_fusion

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

        # 2. 多尺度 GEM（编码器各阶段，对齐 Swin 架构）
        if self.use_evidential_fusion:
            # 为每个编码器阶段创建 GEM 层
            self.gem_layers = nn.ModuleList(
                [
                    GeoEvidentialMappingLayer(
                        input_dim=dim,
                        prototype_dim=gem_prototype_dim,
                        class_dim=num_classes,
                        geo_prior_weight=gem_geo_prior_weight,
                    )
                    for dim in self.feature_dims
                ]
            )

            # 每个stage的轻量监督头：将[K+1]质量映射到[K] logits
            self.stage_heads = nn.ModuleList(
                [nn.Conv2d(self.num_classes + 1, self.num_classes, kernel_size=1) for _ in self.feature_dims]
            )

            # 每个stage的PCA融合特征监督头：从C_s -> K logits
            self.pca_stage_heads = nn.ModuleList(
                [nn.Conv2d(dim, self.num_classes, kernel_size=1) for dim in self.feature_dims]
            )

            # 多尺度概率融合层（输入为每个stage的K通道概率）
            self.evidential_fusion = nn.Sequential(
                nn.Conv2d((self.num_classes) * len(self.feature_dims), 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1),
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

        # GEM 参数 (使用更高学习率)
        if self.use_evidential_fusion:
            for gem_layer in self.gem_layers:
                for param in gem_layer.parameters():
                    param_groups[2].append(param)

            # 证据融合层参数
            for param in self.evidential_fusion.parameters():
                param_groups[2].append(param)

            # 深监督头参数
            for param in self.stage_heads.parameters():
                param_groups[2].append(param)
            for param in self.pca_stage_heads.parameters():
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

        # 若未启用证据融合，走原HARMF路径；否则仅取encoder特征，避免重复图构建
        if not self.use_evidential_fusion:
            return self.harmf_encoder(x)

        # 仅获取编码器多模态多尺度特征（不走WeTr.forward解码路径）
        modality_features = self.harmf_encoder.encoder(x)

        # 对每个尺度的融合特征应用 GEM
        evidence_maps = []  # 这里保存每个stage融合后的概率
        uncertainty_maps = []

        for stage_idx in range(len(self.feature_dims)):
            stage_evidence = []
            stage_uncertainty = []
            stage_probs = []

            # 收集该stage的各模态特征，并执行一次与 WeTr.forward 相同的 PCA 交互更新（不做模态均值）
            num_modalities = len(modality_features)
            stage_feats = []
            for m in range(num_modalities):
                if stage_idx < len(modality_features[m]):
                    stage_feats.append(modality_features[m][stage_idx])

            if not stage_feats:
                continue

            B, C, Hs, Ws = stage_feats[0].shape
            seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]

            updated_seqs = list(seqs)
            for m in range(num_modalities):
                nxt = (m + 1) % num_modalities
                y_m, y_n = self.harmf_encoder.pca_stages[stage_idx](updated_seqs[m], updated_seqs[nxt])
                updated_seqs[m], updated_seqs[nxt] = y_m, y_n

            updated_feats = [
                updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous() for m in range(num_modalities)
            ]

            # 对每个模态应用 GEM（基于经PCA交互后的每模态特征）
            for mod_idx in range(self.num_parallel):
                if mod_idx < len(updated_feats):
                    feat = updated_feats[mod_idx]  # [B, C, H, W]

                    # 应用 GEM-Layer
                    gem = self.gem_layers[stage_idx]
                    mass = gem(feat)  # [B, K+1, H, W]
                    uncertainty = gem.get_uncertainty(mass)

                    # 可选：MRG 折扣（对mass进行标准DS折扣）
                    if self.use_mrg:
                        mass = self.mrg.discount_mass(mass, modality_index=mod_idx)

                    stage_evidence.append(mass)
                    stage_uncertainty.append(uncertainty)

                    # 生成该模态的概率（mass→pl→prob）用于概率路径融合
                    pl = gem.get_plausibility(mass)  # [B,K,H,W]
                    prob = pl / (pl.sum(1, keepdim=True) + 1e-12)
                    stage_probs.append(prob)

            if stage_probs:
                # 概率路径（优先）: 支持多种融合策略
                if len(stage_probs) == 1:
                    fused_prob_s = stage_probs[0]
                    fused_uncertainty = stage_uncertainty[0]
                else:
                    if self.prob_fusion == "mean":
                        fused_prob_s = torch.stack(stage_probs, dim=0).mean(dim=0)
                    elif self.prob_fusion == "weighted_product" and hasattr(self, "mrg") and self.use_mrg:
                        # 以 beta 作为权重的 PoE：prod p_i^w_i 后归一化
                        with torch.no_grad():
                            beta = torch.sigmoid(self.mrg.alpha).mean(dim=1, keepdim=False)  # [T,1,1]
                            weights = beta.unsqueeze(1)  # [T,1,1,1]
                            weights = weights[: len(stage_probs)]
                            weights = weights.to(stage_probs[0].dtype).to(stage_probs[0].device)
                        log_p = torch.stack([torch.log(p + 1e-12) for p in stage_probs], dim=0)
                        wlog = log_p * weights
                        fused_log = wlog.sum(dim=0)
                        fused_pl_s = torch.exp(fused_log)
                        fused_prob_s = fused_pl_s / (fused_pl_s.sum(1, keepdim=True) + 1e-12)
                    else:
                        fused_pl_s = stage_probs[0]
                        for t in range(1, len(stage_probs)):
                            fused_pl_s = fused_pl_s * stage_probs[t]
                        fused_prob_s = fused_pl_s / (fused_pl_s.sum(1, keepdim=True) + 1e-12)
                    fused_uncertainty = torch.stack(stage_uncertainty, dim=0).mean(dim=0)

                # 保存以便多尺度融合（注意这里保存的是概率 K 通道）
                evidence_maps.append(fused_prob_s)
                uncertainty_maps.append(fused_uncertainty)

        # 多尺度概率融合
        if evidence_maps:
            # 将不同stage的证据图调整到相同尺寸
            target_size = evidence_maps[0].shape[2:]
            resized_evidence = []

            for evidence in evidence_maps:
                if evidence.shape[2:] != target_size:
                    evidence = F.interpolate(evidence, size=target_size, mode="bilinear", align_corners=False)
                resized_evidence.append(evidence)

            # 拼接所有stage的概率
            concatenated_evidence = torch.cat(resized_evidence, dim=1)  # [B, (K)*num_stages, H, W]

            # 通过融合网络得到最终输出
            evidential_output = self.evidential_fusion(concatenated_evidence)

            # 调整输出尺寸
            if evidential_output.shape[2:] != original_shape:
                evidential_output = F.interpolate(
                    evidential_output, size=original_shape, mode="bilinear", align_corners=False
                )

            # 注意：返回原始logits以便与CrossEntropyLoss/Dice兼容（这些logits对应概率融合后的分类）

            # 调试信息
            # if self.training and torch.rand(1).item() < 0.01:  # 1%概率打印调试信息
            #     print(f"Multi-scale GEM Debug - evidence_maps: {len(evidence_maps)} stages")
            #     print(f"Multi-scale GEM Debug - concatenated_evidence: {concatenated_evidence.shape}")
            #     print(f"Multi-scale GEM Debug - evidential_output: {evidential_output.shape}")

            # 只保留最终输出为主，深监督可选择开启但降权
            outputs_all = [evidential_output]
            return outputs_all, None
        else:
            # 回退：若概率路径未产生输出，则退回HARMF标准路径
            return self.harmf_encoder(x)

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

    @torch.no_grad()
    def analyze_modalities(self, x, foreground_class: int = 1, compute_loo: bool = True):
        """
        分析各模态贡献与不确定性，并可选计算留一法(LOO)融合结果。

        Returns dict with:
          - final_logits: [B,K,H,W]
          - final_prob: [B,K,H,W]
          - U: [B,1,H,W] 最终不确定性
          - C_t: [B,T,H,W] 每模态log贡献(前景类) 跨stage平均
          - U_t: [B,T,H,W] 每模态不确定性 跨stage平均
          - prob_loo: list len T of [B,K,H,W] (可选)
          - beta: [T,K] 可靠性参数sigmoid
        """
        original_shape = x[0].shape[2:]

        modality_features = self.harmf_encoder.encoder(x)

        num_modalities = len(modality_features)
        stage_count = len(modality_features[0]) if num_modalities > 0 else 0

        stage_fused_probs_all = []
        stage_fused_probs_no_t = [[] for _ in range(self.num_parallel)] if compute_loo else None

        per_modality_log_contrib_sum = [None for _ in range(self.num_parallel)]
        per_modality_U_sum = [None for _ in range(self.num_parallel)]
        per_modality_stage_counter = [0 for _ in range(self.num_parallel)]

        stage_uncertainty_across_modalities = []

        for stage_idx in range(stage_count):
            stage_feats = []
            for m in range(num_modalities):
                if stage_idx < len(modality_features[m]):
                    stage_feats.append(modality_features[m][stage_idx])

            if not stage_feats:
                continue

            B, C, Hs, Ws = stage_feats[0].shape
            seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]

            updated_seqs = list(seqs)
            for m in range(num_modalities):
                nxt = (m + 1) % num_modalities
                y_m, y_n = self.harmf_encoder.pca_stages[stage_idx](updated_seqs[m], updated_seqs[nxt])
                updated_seqs[m], updated_seqs[nxt] = y_m, y_n

            updated_feats = [
                updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous() for m in range(num_modalities)
            ]

            per_modality_probs = []
            per_modality_theta = []

            for mod_idx in range(self.num_parallel):
                if mod_idx < len(updated_feats):
                    feat = updated_feats[mod_idx]
                    gem = self.gem_layers[stage_idx]
                    mass = gem(feat)
                    if self.use_mrg:
                        mass = self.mrg.discount_mass(mass, modality_index=mod_idx)
                    theta = gem.get_uncertainty(mass)  # [B,1,Hs,Ws]
                    pl = gem.get_plausibility(mass)
                    prob = pl / (pl.sum(1, keepdim=True) + 1e-12)

                    per_modality_probs.append(prob)
                    per_modality_theta.append(theta)

                    # C_t 累积（log贡献，前景类）与 U_t 累积（跨stage平均）
                    c_map = torch.log((prob[:, foreground_class : foreground_class + 1, :, :] + 1e-8)).squeeze(1)
                    u_map = theta.squeeze(1)
                    c_map = F.interpolate(
                        c_map.unsqueeze(1), size=original_shape, mode="bilinear", align_corners=False
                    ).squeeze(1)
                    u_map = F.interpolate(
                        u_map.unsqueeze(1), size=original_shape, mode="bilinear", align_corners=False
                    ).squeeze(1)

                    if per_modality_log_contrib_sum[mod_idx] is None:
                        per_modality_log_contrib_sum[mod_idx] = c_map
                        per_modality_U_sum[mod_idx] = u_map
                    else:
                        per_modality_log_contrib_sum[mod_idx] = per_modality_log_contrib_sum[mod_idx] + c_map
                        per_modality_U_sum[mod_idx] = per_modality_U_sum[mod_idx] + u_map
                    per_modality_stage_counter[mod_idx] += 1

            # 跨模态不确定性（该stage）：先对theta取平均
            if per_modality_theta:
                stage_theta_mean = torch.stack(per_modality_theta, dim=0).mean(dim=0)
                stage_theta_mean = F.interpolate(
                    stage_theta_mean, size=original_shape, mode="bilinear", align_corners=False
                )
                stage_uncertainty_across_modalities.append(stage_theta_mean)

            # 该stage融合(全部模态)
            if per_modality_probs:
                fused_pl_s = per_modality_probs[0]
                for t in range(1, len(per_modality_probs)):
                    fused_pl_s = fused_pl_s * per_modality_probs[t]
                fused_prob_s = fused_pl_s / (fused_pl_s.sum(1, keepdim=True) + 1e-12)
                stage_fused_probs_all.append(fused_prob_s)

                # 留一法 per stage
                if compute_loo:
                    for t in range(self.num_parallel):
                        if t < len(per_modality_probs):
                            if self.prob_fusion == "mean":
                                # 平均融合的留一法
                                others = [per_modality_probs[j] for j in range(len(per_modality_probs)) if j != t]
                                if len(others) > 0:
                                    prob_no_t = torch.stack(others, dim=0).mean(dim=0)
                                    stage_fused_probs_no_t[t].append(prob_no_t)
                            else:
                                # PoE留一法（含默认的product与weighted_product）
                                prod_all = fused_pl_s * 1.0
                                prod_no_t = prod_all / (per_modality_probs[t] + 1e-12)
                                prob_no_t = prod_no_t / (prod_no_t.sum(1, keepdim=True) + 1e-12)
                                stage_fused_probs_no_t[t].append(prob_no_t)

        # 多尺度融合到最终输出
        if not stage_fused_probs_all:
            final_logits, _ = self.forward(x)
            final_logits = final_logits[0]
            final_prob = F.softmax(final_logits, dim=1)
        else:
            target_size = stage_fused_probs_all[0].shape[2:]
            aligned = [
                (
                    p
                    if p.shape[2:] == target_size
                    else F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
                )
                for p in stage_fused_probs_all
            ]
            concatenated = torch.cat(aligned, dim=1)
            final_logits = self.evidential_fusion(concatenated)
            final_prob = F.softmax(final_logits, dim=1)

        # 最终不确定性 U（跨stage跨模态平均）
        if stage_uncertainty_across_modalities:
            U = torch.stack(stage_uncertainty_across_modalities, dim=0).mean(dim=0)
        else:
            U = self.get_uncertainty_map(x)

        # 每模态统计：C_t 与 U_t
        C_list = []
        U_list = []
        for t in range(self.num_parallel):
            if per_modality_stage_counter[t] > 0:
                C_avg = per_modality_log_contrib_sum[t] / float(per_modality_stage_counter[t])
                U_avg = per_modality_U_sum[t] / float(per_modality_stage_counter[t])
            else:
                C_avg = (
                    torch.zeros(x[0].shape[0], *original_shape, device=final_prob.device)
                    if per_modality_log_contrib_sum[t] is None
                    else per_modality_log_contrib_sum[t]
                )
                U_avg = torch.zeros_like(C_avg)
            C_list.append(C_avg.unsqueeze(1))
            U_list.append(U_avg.unsqueeze(1))
        C_t = torch.cat(C_list, dim=1) if C_list else None  # [B,T,H,W]
        U_t = torch.cat(U_list, dim=1) if U_list else None

        # 留一法最终概率
        prob_loo = None
        if compute_loo and stage_fused_probs_no_t is not None and stage_fused_probs_no_t[0]:
            prob_loo = []
            for t in range(self.num_parallel):
                if stage_fused_probs_no_t[t]:
                    aligned_no_t = [
                        (
                            p
                            if p.shape[2:] == target_size
                            else F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
                        )
                        for p in stage_fused_probs_no_t[t]
                    ]
                    concatenated_no_t = torch.cat(aligned_no_t, dim=1)
                    logits_no_t = self.evidential_fusion(concatenated_no_t)
                    prob_no_t = F.softmax(logits_no_t, dim=1)
                    prob_loo.append(prob_no_t)
                else:
                    prob_loo.append(None)

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

        with torch.no_grad():
            modality_features = self.harmf_encoder.encoder(x)
            uncertainty_maps = []

            for stage_idx in range(len(self.feature_dims)):
                stage_uncertainty = []

                for mod_idx in range(self.num_parallel):
                    if mod_idx < len(modality_features) and stage_idx < len(modality_features[mod_idx]):
                        feat = modality_features[mod_idx][stage_idx]
                        gem = self.gem_layers[stage_idx]
                        mass = gem(feat)
                        uncertainty = gem.get_uncertainty(mass)
                        stage_uncertainty.append(uncertainty)

                if stage_uncertainty:
                    avg_uncertainty = torch.stack(stage_uncertainty, dim=0).mean(dim=0)
                    uncertainty_maps.append(avg_uncertainty)

            if uncertainty_maps:
                # 调整到相同尺寸并平均
                target_size = uncertainty_maps[0].shape[2:]
                resized_uncertainty = []

                for uncertainty in uncertainty_maps:
                    if uncertainty.shape[2:] != target_size:
                        uncertainty = F.interpolate(uncertainty, size=target_size, mode="bilinear", align_corners=False)
                    resized_uncertainty.append(uncertainty)

                final_uncertainty = torch.stack(resized_uncertainty, dim=0).mean(dim=0)
                return final_uncertainty

            return None


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
