import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSparseGatedCrossAttention(nn.Module):
    def __init__(
        self,
        drug_dim=128,
        prot_dim=1280,
        hidden_dim=128,
        heads=4,
        init_tau=0.5,
        init_alpha=1.0,
    ):
        super().__init__()

        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

        self.site_scorer = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, drug_feat, drug_mask, prot_feat, prot_mask):

        Q = self.drug_proj(drug_feat)
        K = self.prot_proj(prot_feat)
        V = K

        site_logits = self.site_scorer(prot_feat)
        site_logits = site_logits.masked_fill(~prot_mask.unsqueeze(-1), -1e9)

        tau = torch.clamp(self.tau, 0.1, 2.0)
        site_prob = F.softmax(site_logits / tau, dim=1)

        alpha = torch.clamp(self.alpha, 0.0, 5.0)
        scaling = 1.0 + alpha * site_prob

        V_enhanced = V * scaling

        attn_out, _ = self.attn(
            Q, K, V_enhanced,
            key_padding_mask=~prot_mask
        )

        gate = self.gate(attn_out)
        out = gate * attn_out + Q

        out = self.norm(out)

        return out, drug_mask