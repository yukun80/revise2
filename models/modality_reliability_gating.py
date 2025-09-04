import torch
from torch import nn


class ModalityReliabilityGating(nn.Module):
    """
    Modality Reliability Gating (MRG) for 2D inputs.

    Discount per-modality plausibility maps with learnable reliability per (modality, class).

    Shapes:
      - Input pl: [B, K, H, W] per modality
      - Output discounted pl: [B, K, H, W]
    """

    def __init__(self, num_classes: int, num_modalities: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        # alpha -> beta via sigmoid, parameters per modality and class
        # shape [T, K, 1, 1]
        self.alpha = nn.Parameter(torch.zeros(num_modalities, num_classes, 1, 1))

    def forward(self, pl: torch.Tensor, modality_index: int) -> torch.Tensor:
        # pl: [B, K, H, W]
        beta = torch.sigmoid(self.alpha[modality_index])  # [K,1,1]
        # broadcast to [B,K,H,W]
        beta_b = pl.new_ones(pl.size(0), self.num_classes, pl.size(2), pl.size(3)) * beta
        pl_hat = 1.0 - beta_b + beta_b * pl
        return pl_hat

    def discount_mass(self, mass: torch.Tensor, modality_index: int) -> torch.Tensor:
        """
        Discount a Dempster-Shafer mass function using a modality-level reliability r in (0,1).

        DS discounting (scalar r):
          - m'_i     = r * m_i        for singleton classes i
          - m'_Theta = (1 - r) + r * m_Theta

        For stability and consistency, we derive r from per-class alphas by averaging
        sigmoid(alpha) across classes for the given modality.

        Args:
            mass: [B, K+1, H, W]
            modality_index: int
        Returns:
            discounted mass: [B, K+1, H, W]
        """
        B, Kp1, H, W = mass.size()
        K = self.num_classes
        assert Kp1 == K + 1, "Mass tensor must have K+1 channels (including Theta)."

        # reliability scalar r in (0,1) for this modality
        r = torch.sigmoid(self.alpha[modality_index]).mean()
        r = r.to(dtype=mass.dtype, device=mass.device)

        singletons = mass[:, :K, :, :]
        m_theta = mass[:, K : K + 1, :, :]

        discounted_singletons = singletons * r
        discounted_theta = (1.0 - r) + r * m_theta

        return torch.cat([discounted_singletons, discounted_theta], dim=1)

    @staticmethod
    def fuse_discounted_pl(pl_list):
        if not isinstance(pl_list, (list, tuple)) or len(pl_list) == 0:
            raise ValueError("pl_list must be non-empty")
        fused = pl_list[0]
        for t in range(1, len(pl_list)):
            fused = fused * pl_list[t]
        return fused

    @staticmethod
    def normalize_to_prob(fused_pl: torch.Tensor) -> torch.Tensor:
        K_sum = fused_pl.sum(1, keepdim=True)
        return fused_pl / (K_sum + 1e-12)
