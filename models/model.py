import torch.nn as nn
from models.drug_encoder import DrugEGNN
from models.interaction import LigandAwareCrossAttention
from models.predictor import AffinityPredictor


class DTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = DrugEGNN(edge_dim=3)
        self.interaction = LigandAwareCrossAttention()
        self.predictor = AffinityPredictor()

    def forward(self, drug_graph, prot_feat, prot_mask):
        drug_feat, drug_mask = self.drug_encoder(drug_graph)

        # 🔥 接收新增的全局蛋白特征
        fused_drug, drug_mask, global_prot = self.interaction(
            drug_feat, drug_mask,
            prot_feat, prot_mask
        )

        # 🔥 把全局蛋白特征一起传给打分器
        return self.predictor(fused_drug, drug_mask, global_prot)
