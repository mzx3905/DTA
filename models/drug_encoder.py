import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class EGNNLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        # 🔥 补回等变图的灵魂：3D坐标微调网络
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, 1, bias=False)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(out_dim + in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, coords, edge_attr):
        row, col = edge_index
        diff = coords[row] - coords[col]
        dist = (diff ** 2).sum(dim=1, keepdim=True)

        # 计算边特征
        edge_feat = self.edge_mlp(
            torch.cat([x[row], x[col], edge_attr, dist], dim=1)
        )

        # 更新 3D 坐标 (坐标永远是 3 维的，所以 zeros_like(coords) 没问题)
        coord_diff = diff / (dist.sqrt() + 1e-8)
        coord_update = coord_diff * self.coord_mlp(edge_feat)
        coords_agg = torch.zeros_like(coords)
        coords_agg.index_add_(0, row, coord_update)
        coords = coords + coords_agg

        # ==========================================
        # 🔥 绝杀修复：不能用 zeros_like(x)！
        # 必须根据 batch_size (x.shape[0]) 和 边特征的高级维度 (edge_feat.shape[1])
        # 动态开辟一个 [1029, 128] 的正确邮箱！
        # ==========================================
        agg = torch.zeros(x.size(0), edge_feat.size(1), dtype=x.dtype, device=x.device)

        agg.index_add_(0, row, edge_feat)
        x = self.node_mlp(torch.cat([x, agg], dim=1))

        return x, coords


class DrugEGNN(nn.Module):
    def __init__(self, in_dim=78, edge_dim=10, hidden_dim=128, num_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            EGNNLayer(dims[i], edge_dim, dims[i + 1])
            for i in range(num_layers)
        ])

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        coords = graph.coords
        edge_attr = graph.edge_attr
        batch = graph.batch

        # 逐层更新特征和坐标
        for layer in self.layers:
            x, coords = layer(x, edge_index, coords, edge_attr)

        x_dense, mask = to_dense_batch(x, batch)
        return x_dense, mask
