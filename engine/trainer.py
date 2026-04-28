import torch
import torch.nn as nn


def train(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0

    for drug_graph, prot_feat, prot_mask, y in loader:

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