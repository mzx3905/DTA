import torch
import torch.nn as nn
import torch.nn.functional as F


class LigandAwareCrossAttention(nn.Module):
    def __init__(
            self,
            drug_dim=128,
            prot_dim=1280,
            hidden_dim=128,
            heads=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. 强制模态对齐（绝对不能删）
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),  # 暖启动更是必须防死记硬背
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 2. 交叉注意力：自带配体感知的口袋发现器
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

        # 3. 动态门控（基于交互后的上下文，而不是静态蛋白）
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, drug_feat, drug_mask, prot_feat, prot_mask):
        Q = self.drug_proj(drug_feat)
        K = self.prot_proj(prot_feat) # [B, L, 128]
        V = K

        attn_out, attn_weights = self.attn(
            Q, K, V,
            key_padding_mask=~prot_mask
        )

        gate = self.gate(attn_out)
        out = gate * attn_out + Q
        out = self.norm(out)

        # 🔥 绝杀修复：提取蛋白质全局身份特征，别再把它扔了！
        # 排除 Padding 噪音，只对有效氨基酸求均值
        K_masked = K.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
        valid_lens = prot_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_prot = K_masked.sum(dim=1) / valid_lens # [B, 128]

        # 返回三个参数！
        return out, drug_mask, global_prot
