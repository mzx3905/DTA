import torch
import torch.nn as nn


class AffinityPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.pool = nn.Linear(hidden_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, feat, mask):
        logits = self.pool(feat)  # [B, N, 1]

        # 🔥 必须在 Softmax 之前，用 -1e9 彻底处决 Padding 噪音！
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)

        # 此时参与 Softmax 的假原子分母为 e^(-1e9) ≈ 0，完美绝缘
        weights = torch.softmax(logits, dim=1)

        pooled = (feat * weights).sum(dim=1)
        return self.mlp(pooled).squeeze(-1)