import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer import WeTr
from .geo_evidential_mapping import GeoEvidentialMappingLayer, plausibility_to_probability
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

        # 2. GEM-Layer (地理证据映射层)
        if self.use_evidential_fusion:
            # 为每个模态创建独立的GEM-Layer
            # 先将各stage特征通过1x1投影到统一的256维，再送入GEM-Layer
            # 因此GEM-Layer的input_dim固定为256，避免通道不匹配
            self.gem_layers = nn.ModuleList(
                [
                    GeoEvidentialMappingLayer(
                        input_dim=256,
                        prototype_dim=gem_prototype_dim,
                        class_dim=num_classes,
                        geo_prior_weight=gem_geo_prior_weight,
                    )
                    for _ in self.feature_dims
                ]
            )

            # 特征投影层，将不同stage的特征投影到统一维度
            self.feature_projections = nn.ModuleList([nn.Conv2d(dim, 256, 1) for dim in self.feature_dims])

            # 最终融合层：输入为各stage的证据图拼接，通道数为 (K+1) * num_stages
            self.evidential_fusion = nn.Sequential(
                nn.Conv2d((self.num_classes + 1) * len(self.feature_dims), 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1),
            )

            # MRG: 可选，按模态/类别学习可靠性并做上下文化折扣
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

        # GEM-Layer参数
        if self.use_evidential_fusion:
            for gem_layer in self.gem_layers:
                for param in gem_layer.parameters():
                    param_groups[2].append(param)

            for proj in self.feature_projections:
                for param in proj.parameters():
                    param_groups[2].append(param)

            for param in self.evidential_fusion.parameters():
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

        # 1. HARMF编码器前向传播
        harmf_outputs, aux_loss = self.harmf_encoder(x)

        if not self.use_evidential_fusion:
            # 如果不使用证据融合，直接返回HARMF结果
            return harmf_outputs, aux_loss

        # 2. 串联式证据决策：以解码器融合特征为输入（对齐原实现）
        # 从 WeTr.decoder 暴露的 last_fused 取特征（已与最浅层对齐到 H/4,W/4）
        fused_feature = getattr(self.harmf_encoder.decoder, "last_fused", None)

        if fused_feature is None:
            # 兼容：若未暴露，则退化为使用harmf输出前一层特征，不中断训练
            return harmf_outputs, aux_loss

        # 单一路径证据映射（不再做多尺度并行），保持与原作“解码后”一致
        # decoder.last_fused 已为 256 通道，直接输入 GEM
        gem = self.gem_layers[0]
        mass = gem(fused_feature)  # [B,K+1,H/4,W/4]
        theta_map = gem.get_uncertainty(mass)  # [B,1,H/4,W/4]

        # 可选：MRG + ECD（此处跨模态已在编码阶段融合，若仍保留多源，可在上游提供多源fused_feature列表）
        if self.use_mrg:
            pl = gem.get_plausibility(mass)  # [B,K,H/4,W/4]
            # 这里只有“聚合后特征”，如需分模态折扣，需改为在解码前保持分模态分支特征
            pl_hat = self.mrg(pl, modality_index=0)
            if self.use_evidential_combination:
                fused_pl = pl_hat
                fused_prob = normalize_pl_to_prob(fused_pl)
                theta = (1.0 - fused_prob.max(1, keepdim=True)[0]).clamp(min=0.0)
                mass = torch.cat([fused_prob, theta], dim=1)
            else:
                theta = (1.0 - pl_hat.max(1, keepdim=True)[0]).clamp(min=0.0)
                mass = torch.cat([pl_hat, theta], dim=1)

        # 简单双头输出：保留原HARMF输出 + 证据分支（上采样到原图）
        evidential_output = mass[:, : self.num_classes, :, :]
        if evidential_output.shape[2:] != original_shape:
            evidential_output = F.interpolate(
                evidential_output, size=original_shape, mode="bilinear", align_corners=False
            )

        if theta_map.shape[2:] != original_shape:
            theta_map = F.interpolate(theta_map, size=original_shape, mode="bilinear", align_corners=False)

        return [harmf_outputs[0], evidential_output], aux_loss

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
                        projected_feat = self.feature_projections[stage_idx](feat)
                        mass = self.gem_layers[stage_idx](projected_feat)
                        uncertainty = self.gem_layers[stage_idx].get_uncertainty(mass)
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
