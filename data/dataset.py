import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

MAX_PROT_LEN = 1022


class DTADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):

    drug_graphs = Batch.from_data_list([b["drug_graph"] for b in batch])

    prot_feats = []
    prot_masks = []

    for b in batch:
        feat = b["prot_feat"][:MAX_PROT_LEN]
        mask = torch.ones(len(feat), dtype=torch.bool)

        prot_feats.append(feat)
        prot_masks.append(mask)

    prot_feats = pad_sequence(prot_feats, batch_first=True)
    prot_masks = pad_sequence(prot_masks, batch_first=True)

    labels = torch.tensor([b["y"] for b in batch], dtype=torch.float)

    return drug_graphs, prot_feats, prot_masks, labels