import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.nn.utils.rnn import pad_sequence

MAX_PROT_LEN = 1022

# 🔥 绝对保命符：防止 PyTorch 后台多线程与 DataLoader 子进程互相抢夺 CPU 导致死锁卡死
torch.set_num_threads(1)

class RealDTADataset(Dataset):
    def __init__(self, data_pairs, drug_dir, prot_dir):
        self.data_pairs = data_pairs
        self.drug_dir = drug_dir
        self.prot_dir = prot_dir

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        drug_id, target_id, y = self.data_pairs[idx]

        # ==========================================
        # 1. 动态探测并加载药物图 (精准破解 5 元素黑盒)
        # ==========================================
        drug_path = os.path.join(self.drug_dir, f"{drug_id}.npy")
        loaded_array = np.load(drug_path, allow_pickle=True)

        if isinstance(loaded_array, np.ndarray):
            if loaded_array.size == 1:
                loaded_data = loaded_array.item()
            else:
                loaded_data = loaded_array.tolist()
        else:
            loaded_data = loaded_array

        raw_x, raw_edge_index, raw_edge_attr, raw_coords = None, None, None, None

        if isinstance(loaded_data, dict):
            raw_x = loaded_data.get('x')
            raw_edge_index = loaded_data.get('edge_index')
            raw_edge_attr = loaded_data.get('edge_attr')
            raw_coords = loaded_data.get('coords', loaded_data.get('pos'))

        elif isinstance(loaded_data, (list, tuple)):
            # 🔥 绝杀修复点：判断第0个元素是不是数字(c_size)！
            if isinstance(loaded_data[0], (int, float, np.integer, np.floating)):
                # 它是原作者奇葩的 5 元素格式: [c_size, features, edge_index, coords, edge_weight]
                raw_x = loaded_data[1]
                raw_edge_index = loaded_data[2]
                raw_coords = loaded_data[3] if len(loaded_data) > 3 else None
                raw_edge_attr = loaded_data[4] if len(loaded_data) > 4 else None
            else:
                # 它是正常的 4 元素格式: [features, edge_index, edge_attr, coords]
                raw_x = loaded_data[0]
                raw_edge_index = loaded_data[1]
                raw_edge_attr = loaded_data[2] if len(loaded_data) > 2 else None
                raw_coords = loaded_data[3] if len(loaded_data) > 3 else None

        elif hasattr(loaded_data, 'x') and hasattr(loaded_data, 'edge_index'):
            raw_x = getattr(loaded_data, 'x', None)
            raw_edge_index = getattr(loaded_data, 'edge_index', None)
            raw_edge_attr = getattr(loaded_data, 'edge_attr', None)
            raw_coords = getattr(loaded_data, 'coords', getattr(loaded_data, 'pos', None))
        else:
            raise ValueError(f"彻底无法识别的药物图底层格式！")

        # ----------------------------------------------------
        # 重新组装纯净版 Data，并修复边索引维度
        # ----------------------------------------------------
        x = torch.as_tensor(raw_x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        # 🔥 绝杀修复点：处理 PyG 的 [2, num_edges] 格式要求
        if raw_edge_index is not None and len(raw_edge_index) > 0:
            edge_index = torch.as_tensor(raw_edge_index, dtype=torch.long)
            # 如果是 [num_edges, 2] 格式，强制转置为 [2, num_edges]
            if edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        drug_graph = Data(x=x, edge_index=edge_index)

        if raw_edge_attr is not None:
            drug_graph.edge_attr = torch.as_tensor(raw_edge_attr, dtype=torch.float32)

        if raw_coords is not None:
            drug_graph.coords = torch.as_tensor(raw_coords, dtype=torch.float32)
        else:
            drug_graph.coords = torch.zeros((x.shape[0], 3), dtype=torch.float32)

        # ==========================================
        # 2. 加载蛋白特征 (保持原样)
        # ==========================================
        prot_path = os.path.join(self.prot_dir, f"{target_id}.npy")
        prot_feat = torch.from_numpy(np.load(prot_path)).float()

        return {
            "drug_graph": drug_graph,
            "prot_feat": prot_feat,
            "y": float(y)
        }

def collate_fn(batch):
    drug_graphs = Batch.from_data_list([b["drug_graph"] for b in batch])

    prot_feats = []
    prot_masks = []
    ys = []

    for b in batch:
        feat = b["prot_feat"][:MAX_PROT_LEN]
        prot_feats.append(feat)
        prot_masks.append(torch.ones(len(feat), dtype=torch.bool))
        ys.append(b["y"])

    prot_feats = pad_sequence(prot_feats, batch_first=True)
    prot_masks = pad_sequence(prot_masks, batch_first=True)

    return drug_graphs, prot_feats, prot_masks, torch.tensor(ys, dtype=torch.float32)
