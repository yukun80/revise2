import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer import WeTr
from .geo_evidential_mapping import GeoEvidentialMappingLayer
from .modality_reliability_gating import ModalityReliabilityGating
from .evidential_combination import fuse_discounted_plausibilities, normalize_pl_to_prob


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
        use_evidential_combination=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_parallel = num_parallel
        self.use_evidential_fusion = use_evidential_fusion
        self.use_mrg = use_mrg
        self.use_evidential_combination = use_evidential_combination

        print("-----------------HAEF-Net Params--------------------------------------")
        print("backbone:", backbone)
        print("num_classes:", num_classes)
        print("num_modalities:", num_parallel)
        print("use_evidential_fusion:", use_evidential_fusion)
        print("gem_prototype_dim:", gem_prototype_dim)
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

    def _dempster_combination(self, evidence_list):
        """
        使用Dempster组合规则融合多个证据质量函数

        Args:
            evidence_list: List of [B, K+1, H, W] mass functions
        Returns:
            combined_evidence: [B, K+1, H, W] 组合后的质量函数
        """
        if len(evidence_list) == 1:
            return evidence_list[0]

        # 初始化组合结果
        combined = evidence_list[0]

        # 逐个组合证据
        for evidence in evidence_list[1:]:
            combined = self._dempster_combine_two(combined, evidence)

        return combined

    def _dempster_combine_two(self, m1, m2):
        """
        使用Dempster组合规则组合两个质量函数

        Args:
            m1, m2: [B, K+1, H, W] mass functions
        Returns:
            combined: [B, K+1, H, W] 组合后的质量函数
        """
        B, K_plus_1, H, W = m1.shape
        K = K_plus_1 - 1  # 类别数
        m1_theta = m1[:, -1:, :, :]  # [B, 1, H, W]
        m2_theta = m2[:, -1:, :, :]  # [B, 1, H, W]
        m1_singletons = m1[:, :-1, :, :]  # [B, K, H, W]
        m2_singletons = m2[:, :-1, :, :]  # [B, K, H, W]

        # 冲突 K = Σ_{i≠j} m1({i}) m2({j})
        sum_m1 = m1_singletons.sum(dim=1, keepdim=True)  # [B,1,H,W]
        sum_m2 = m2_singletons.sum(dim=1, keepdim=True)  # [B,1,H,W]
        diag = (m1_singletons * m2_singletons).sum(dim=1, keepdim=True)  # [B,1,H,W]
        conflict = (sum_m1 * sum_m2) - diag  # [B,1,H,W]

        # 归一化分母 1 - K，数值稳定
        denom = (1.0 - conflict).clamp(min=1e-6)

        # 组合单例（向量化）：m({k}) = [m1_k*m2_k + m1_k*m2_Θ + m1_Θ*m2_k] / (1-K)
        numer_singletons = (
            m1_singletons * m2_singletons + m1_singletons * m2_theta + m1_theta * m2_singletons
        )  # [B,K,H,W]
        combined_singletons = numer_singletons / denom

        # 组合 Theta：m(Θ) = [m1(Θ) * m2(Θ)] / (1-K)
        numer_theta = m1_theta * m2_theta  # [B,1,H,W]
        combined_theta = numer_theta / denom

        combined = torch.cat([combined_singletons, combined_theta], dim=1)
        # 保险起见再次归一化
        combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-12)

        return combined

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
                # 概率路径（优先）: 按模态乘积并归一化
                if len(stage_probs) == 1:
                    fused_prob_s = stage_probs[0]
                    fused_uncertainty = stage_uncertainty[0]
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
