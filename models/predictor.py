import torch
import torch.nn as nn


class AffinityPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.pool = nn.Linear(hidden_dim, 1)

        # 🔥 核心修正：输入维度翻倍 (128 -> 256)，因为要接收拼接特征！
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, drug_feat, drug_mask, global_prot):
        # 1. 浓缩药物特征
        logits = self.pool(drug_feat)  # [B, N, 1]
        logits = logits.masked_fill(~drug_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled_drug = (drug_feat * weights).sum(dim=1)  # [B, 128]

        # 2. 🔥 绝杀动作：将药物与全局靶点特征拼接 (Concatenation)
        # [B, 128] 和 [B, 128] 拼接成 [B, 256]
        joint_representation = torch.cat([pooled_drug, global_prot], dim=1)

        # 3. 最终预测
        return self.mlp(joint_representation).squeeze(-1)
