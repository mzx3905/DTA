import torch
import torch.nn as nn
from utils.metrics import model_evaluate


def train(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0
    for drug_graph, prot_feat, prot_mask, y in loader:
        # 数据推入设备
        drug_graph = drug_graph.to(device)
        prot_feat = prot_feat.to(device)
        prot_mask = prot_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(drug_graph, prot_feat, prot_mask)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_model(model, loader, device, dataset_name):  # 🔥 关键修复点：这里必须有 4 个参数！
    model.eval()
    y_true = []
    y_pred = []

    for drug_graph, prot_feat, prot_mask, y in loader:
        drug_graph = drug_graph.to(device)
        prot_feat = prot_feat.to(device)
        prot_mask = prot_mask.to(device)

        pred = model(drug_graph, prot_feat, prot_mask)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    # 将真实的预测值交给我们新写的全指标评估函数
    return model_evaluate(y_true, y_pred, dataset=dataset_name)
