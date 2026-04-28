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
        # 🔥 补上被他删掉的坐标偏移网络
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, 1, bias=False)  # 输出 1 维的坐标标量
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

        edge_feat = self.edge_mlp(
            torch.cat([x[row], x[col], edge_attr, dist], dim=1)
        )

        # 🔥 补充：更新三维坐标 (动态空间自适应)
        coord_diff = diff / (dist.sqrt() + 1e-8)
        coord_update = coord_diff * self.coord_mlp(edge_feat)
        coords_agg = torch.zeros_like(coords)
        coords_agg.index_add_(0, row, coord_update)
        coords = coords + coords_agg  # 得到了微调后的新 3D 坐标！

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, edge_feat)
        x = self.node_mlp(torch.cat([x, agg], dim=1))

        # 必须把更新后的 coords 传给下一层！
        return x, coords


class DrugEGNN(nn.Module):
    def __init__(self, in_dim=78, edge_dim=10, hidden_dim=128, num_layers=3):
        super().__init__()

        dims = [in_dim] + [hidden_dim] * num_layers

        self.layers = nn.ModuleList([
            EGNNLayer(dims[i], edge_dim, dims[i+1])
            for i in range(num_layers)
        ])

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        coords = graph.coords
        edge_attr = graph.edge_attr
        batch = graph.batch

        for layer in self.layers:
            x = layer(x, edge_index, coords, edge_attr)

        x_dense, mask = to_dense_batch(x, batch)

        return x_dense, mask