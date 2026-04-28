import torch.nn as nn
from models.drug_encoder import DrugEGNN
from models.interaction import StableSparseGatedCrossAttention
from models.predictor import AffinityPredictor


class DTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = DrugEGNN(edge_dim=3)
        self.interaction = StableSparseGatedCrossAttention()
        self.predictor = AffinityPredictor()

    def forward(self, drug_graph, prot_feat, prot_mask):
        drug_feat, drug_mask = self.drug_encoder(drug_graph)

        fused_feat, mask = self.interaction(
            drug_feat, drug_mask,
            prot_feat, prot_mask
        )

        return self.predictor(fused_feat, mask)
