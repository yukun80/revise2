import torch
import torch.nn as nn


class Mlp2(nn.Module):
    """Lightweight MLP used by the relation discriminator.

    This mirrors the small MLP used in `models/mix_transformer.py` (Mlp_2).
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PixelCrossAttention(nn.Module):
    """Linear-complexity, pixel-wise cross-attention for GeminiFusion.

    Key idea:
    - Re-interpret tokens as the batch axis for attention, and keep sequence length tiny (S=2):
      one slot for a learnable noise token and one for the paired, relation-modulated token.
      This yields linear complexity with respect to number of tokens.
    - Compute attention twice per batch: modality A queries B, and modality B queries A.
    - Use a relation discriminator to modulate the cross-keys, and inject layer-adaptive noise
      into keys/values to stabilize per-token attention.

    Shapes:
    - Inputs: x0, x1 are (B, N, C)
    - Internally for each batch element, we feed MHA with q of shape (1, N, C) and
      k, v of shape (2, N, C). With batch_first=False, MultiheadAttention interprets
      shape as (S, N, E), where we treat N as the token index. This provides per-token
      independent attention across a length-2 sequence, achieving linear complexity.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Two directional pixel-wise cross attentions
        self.cross_attn_0_to_1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
        self.cross_attn_1_to_0 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)

        # Relation discriminator: concatenate per-pixel tokens from two modalities -> gate vector
        self.relation_judger = nn.Sequential(
            Mlp2(dim * 2, dim, dim),
            nn.Softmax(dim=-1),
        )

        # Layer-adaptive noise (two directions) for keys and values
        self.k_noise = nn.Embedding(2, dim)
        self.v_noise = nn.Embedding(2, dim)

    @torch.no_grad()
    def _expand_noise_like(self, noise_vector: torch.Tensor, q_like: torch.Tensor) -> torch.Tensor:
        """Broadcast a (C,) noise vector to the shape (1, N, C) like q.

        q_like is (1, N, C). noise_vector is (C,). Returns (1, N, C).
        """
        # [C] -> [1, 1, C] -> [1, N, C]
        return noise_vector.unsqueeze(0).unsqueeze(0).expand(q_like.size(0), q_like.size(1), -1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x0.shape
        assert x1.shape == (B, N, C)

        new_x0 = []
        new_x1 = []

        for b in range(B):
            # 0 -> 1
            q_01 = x0[b].unsqueeze(0)  # (1, N, C)
            judge_in_01 = torch.cat([x0[b].unsqueeze(0), x1[b].unsqueeze(0)], dim=-1)  # (1, N, 2C)
            relation_01 = self.relation_judger(judge_in_01)  # (1, N, C), soft selection

            noise_k_01 = self._expand_noise_like(self.k_noise.weight[0], q_01) + q_01
            noise_v_01 = self._expand_noise_like(self.v_noise.weight[0], q_01) + q_01

            k_01 = torch.cat([noise_k_01, q_01 * relation_01], dim=0)  # (2, N, C)
            v_01 = torch.cat([noise_v_01, x1[b].unsqueeze(0)], dim=0)   # (2, N, C)

            out_01, _ = self.cross_attn_0_to_1(q_01, k_01, v_01)
            new_x0.append(x0[b] + out_01.squeeze(0))

            # 1 -> 0
            q_10 = x1[b].unsqueeze(0)  # (1, N, C)
            judge_in_10 = torch.cat([x1[b].unsqueeze(0), x0[b].unsqueeze(0)], dim=-1)
            relation_10 = self.relation_judger(judge_in_10)

            noise_k_10 = self._expand_noise_like(self.k_noise.weight[1], q_10) + q_10
            noise_v_10 = self._expand_noise_like(self.v_noise.weight[1], q_10) + q_10

            k_10 = torch.cat([noise_k_10, q_10 * relation_10], dim=0)
            v_10 = torch.cat([noise_v_10, x0[b].unsqueeze(0)], dim=0)

            out_10, _ = self.cross_attn_1_to_0(q_10, k_10, v_10)
            new_x1.append(x1[b] + out_10.squeeze(0))

        new_x0 = torch.stack(new_x0, dim=0)
        new_x1 = torch.stack(new_x1, dim=0)
        return new_x0, new_x1


__all__ = [
    "PixelCrossAttention",
]



if __name__ == "__main__":
    # Simple sanity check
    torch.manual_seed(0)
    B, N, C = 2, 16, 64

    x0 = torch.randn(B, N, C)
    x1 = torch.randn(B, N, C)

    attn = PixelCrossAttention(dim=C, num_heads=8, dropout=0.0)
    y0, y1 = attn(x0, x1)

    print("Input shapes:", x0.shape, x1.shape)
    print("Output shapes:", y0.shape, y1.shape)
