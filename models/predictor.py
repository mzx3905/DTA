import torch
import torch.nn as nn


class AffinityPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.pool = nn.Linear(hidden_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, feat, mask):
        logits = self.pool(feat)  # [B, N, 1]

        # 🔥 必须先彻底处决 Padding 噪音，再做 Softmax!
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(logits, dim=1)

        pooled = (feat * weights).sum(dim=1)
        return self.mlp(pooled).squeeze(-1)
