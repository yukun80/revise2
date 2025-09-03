import torch


def fuse_discounted_plausibilities(pl_list):
    if not isinstance(pl_list, (list, tuple)) or len(pl_list) == 0:
        raise ValueError("pl_list must be non-empty")
    fused = pl_list[0]
    for t in range(1, len(pl_list)):
        fused = fused * pl_list[t]
    return fused


def normalize_pl_to_prob(fused_pl: torch.Tensor) -> torch.Tensor:
    K_sum = fused_pl.sum(1, keepdim=True)
    return fused_pl / (K_sum + 1e-12)
