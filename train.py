import os
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader

from data.dataset import RealDTADataset, collate_fn
# 🔥 唯一入口：必须确保你的 engine/trainer.py 里有 4个参数的 eval_model
from engine.trainer import train, eval_model
from models.model import DTAModel

# ==========================================
# 数据加载组装逻辑
# ==========================================
def load_real_data_pairs(dataset_name="davis"):
    print(f"正在扫描 [{dataset_name.upper()}] 数据集配对信息...")
    base_dir = f'./source/data/{dataset_name}'

    # 1. 加载字典
    drugs = json.load(open(f'{base_dir}/drugs.txt'), object_pairs_hook=OrderedDict)
    targets = json.load(open(f'{base_dir}/targets.txt'), object_pairs_hook=OrderedDict)

    drug_keys = list(drugs.keys())
    target_keys = list(targets.keys())

    # 2. 读取亲和力矩阵
    try:
        with open(f'{base_dir}/affinities', 'rb') as f:
            affinity_matrix = pickle.load(f, encoding='latin1')
    except:
        try:
            affinity_matrix = np.loadtxt(f'{base_dir}/affinities', delimiter=',')
        except:
            affinity_matrix = pickle.load(open(f'{base_dir}/affinities', 'rb'), encoding='latin1')

    # 🔥 绝杀修复点：取消错误的缩进！必须与上面的 try 块平齐！
    # 无论前面怎么读出来的，只要是 davis，就必须做对数降维！
    if dataset_name == 'davis':
        print(">> 触发 DAVIS 数据集专属数学降维: -np.log10(Y / 1e9)")
        affinity_matrix = -np.log10(affinity_matrix / 1e9)

    drug_dir = f'./new_train/drug_graphs/{dataset_name}'
    prot_dir = f'./new_train/protein_esm2_features/{dataset_name}'

    data_pairs = []
    missing_count = 0

    for i in range(len(drug_keys)):
        for j in range(len(target_keys)):
            y_val = affinity_matrix[i][j]
            if not np.isnan(y_val):
                d_id = drug_keys[i]
                t_id = target_keys[j]

                if os.path.exists(f"{drug_dir}/{d_id}.npy") and os.path.exists(f"{prot_dir}/{t_id}.npy"):
                    data_pairs.append((d_id, t_id, float(y_val)))
                else:
                    missing_count += 1

    print(f"✅ 构建完成！有效数据对: {len(data_pairs)} 个。已过滤缺失文件: {missing_count} 个。")
    return data_pairs, drug_dir, prot_dir

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 🔥 1. 全局训练配置
    dataset_name = "davis"  # 填 "davis" 或 "kiba"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    max_epochs = 1000
    patience = 100

    checkpoint_dir = f'./checkpoint/{dataset_name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = open(f'{checkpoint_dir}/training_log.txt', 'w')

    # 🔥 2. 准备数据与划分
    data_pairs, drug_dir, prot_dir = load_real_data_pairs(dataset_name)

    np.random.seed(42)
    np.random.shuffle(data_pairs)
    train_size = int(0.8 * len(data_pairs))

    train_pairs = data_pairs[:train_size]
    val_pairs = data_pairs[train_size:]

    train_dataset = RealDTADataset(train_pairs, drug_dir, prot_dir)
    val_dataset = RealDTADataset(val_pairs, drug_dir, prot_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=3
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=3
    )

    # 🔥 3. 初始化模型与优化器
    model = DTAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) # 甚至可以尝试 1e-2
    # 监听 Val MSE，如果 15 轮没有下降，学习率砍半，强制模型进行微调而不是震荡
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )


    # 🔥 4. 炼丹循环
    print(f"\n🚀 引擎点火！正在 {device} 上启动带早停机制的训练...")
    best_val_mse = float('inf')
    stop_counter = 0

    for epoch in range(1, max_epochs + 1):
        # 训练阶段
        train_mse = train(model, train_loader, optimizer, device)

        # 评估阶段 (由于删除了本地的废弃函数，它现在会去调用 engine.trainer 里的正确函数)
        r = eval_model(model, val_loader, device, dataset_name)
        val_mse, val_ci, val_rm2, val_pearson = r[0], r[1], r[2], r[3]

        # 日志记录
        log_str = f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | CI: {val_ci:.4f} | RM2: {val_rm2:.4f} | Pearson: {val_pearson:.4f}"
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()
        # 🔥 必须在这里取消注释！把实际算出来的 val_mse 传进去！
        scheduler.step(val_mse)

        # 早停与保存
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            stop_counter = 0
            model_path = f"{checkpoint_dir}/best_model.pkl"
            torch.save(model.state_dict(), model_path)
            print(f"   🌟 发现更优验证集 MSE: {best_val_mse:.4f}，模型参数已保存！")
        else:
            stop_counter += 1

        if stop_counter >= patience:
            msg = f"\n🛑 [Early Stopping] 连续 {patience} 轮未提升，提前结束训练。最佳 Val MSE = {best_val_mse:.4f}"
            print(msg)
            log_file.write(msg + "\n")
            break

    log_file.close()
