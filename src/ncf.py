"""E5：NCF 风格的显式评分预测模型（PyTorch）。

**定位很重要**：这不是原论文 Neural Collaborative Filtering 的 NeuMF 复现。
- 原论文面向**隐式反馈**（0/1），用 BPR / 负采样 + sigmoid 输出；
- 这里面向 **1–5 星显式评分**，用 MSE 回归，只保留“Embedding 拼接 + MLP”的骨架。
报告中一律写作“NCF 风格的显式评分预测模型”。

本模块只在 `scripts/run_e5.py`（conda 环境 deeplearning）中使用，
不会被 app.py 导入，因此基础环境（无 torch）仍可正常运行课堂演示。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    # 默认值 = 仅在 fold 1 的**训练集内部验证集**上比较 4 组配置后选出的最优组合
    # （选参过程见 docs/experiment_report.md；测试集从未参与选择）
    emb_dim: int = 32          # 用户 / 电影各自的 embedding 维度，拼接后 64
    hidden: tuple[int, ...] = (64, 32)
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 6          # 连续多少轮验证损失不降就停
    val_ratio: float = 0.1     # 仅从训练集内部切分
    seed: int = 42
    device: str = "cuda"


class NCFMLP(nn.Module):
    """Embedding(u) ⊕ Embedding(i) → MLP → 标量评分。"""

    def __init__(self, n_users: int, n_items: int, cfg: TrainConfig):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, cfg.emb_dim)
        self.item_emb = nn.Embedding(n_items, cfg.emb_dim)
        layers: list[nn.Module] = []
        dim = cfg.emb_dim * 2
        for h in cfg.hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self.mlp = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=1)
        return self.mlp(x).squeeze(1)


def _batch_loss(model: NCFMLP, loss_fn, u, i, r, idx, cfg: TrainConfig, train: bool, opt=None) -> float:
    """在一个 mini-batch 上算损失（train=True 时反传）。返回样本加权损失。"""
    bu, bi, br = u[idx], i[idx], r[idx]
    pred = model(bu, bi)
    loss = loss_fn(pred, br)
    if train:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss.item()) * len(idx)


def train_fold(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    cfg: TrainConfig,
    seed_offset: int = 0,
) -> dict:
    """在单折训练集上训练；验证集从训练集内部随机切分，测试集不参与。"""
    torch.manual_seed(cfg.seed + seed_offset)
    np.random.seed(cfg.seed + seed_offset)
    rng = np.random.default_rng(cfg.seed + seed_offset)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    n = len(ratings)
    n_val = max(1, int(round(n * cfg.val_ratio)))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    u = torch.as_tensor(user_idx, dtype=torch.long, device=device)
    i = torch.as_tensor(item_idx, dtype=torch.long, device=device)
    r = torch.as_tensor(ratings, dtype=torch.float32, device=device)

    model = NCFMLP(n_users, n_items, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    t0 = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(tr_idx))
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = tr_idx[order[start:start + cfg.batch_size]]
            total += _batch_loss(model, loss_fn, u, i, r, idx, cfg, train=True, opt=opt)
        train_loss = total / len(tr_idx)

        model.eval()
        with torch.no_grad():
            val_loss = _batch_loss(model, loss_fn, u, i, r, val_idx, cfg, train=False) / len(val_idx)

        history.append({
            "epoch": epoch,
            "train_rmse": float(np.sqrt(train_loss)),
            "val_rmse": float(np.sqrt(val_loss)),
        })

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_rmse": float(np.sqrt(best_val)),
        "train_seconds": time.perf_counter() - t0,
        "device": str(device),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
    }


@torch.no_grad()
def predict(model: NCFMLP, user_idx: np.ndarray, item_idx: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """批量预测并裁剪到 [1,5]，与 CF / MF 的评估口径一致。"""
    device = next(model.parameters()).device
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(user_idx), batch_size):
        bu = torch.as_tensor(user_idx[start:start + batch_size], dtype=torch.long, device=device)
        bi = torch.as_tensor(item_idx[start:start + batch_size], dtype=torch.long, device=device)
        out.append(model(bu, bi).detach().cpu().numpy())
    preds = np.concatenate(out) if out else np.empty(0, dtype=np.float32)
    return np.clip(preds, 1.0, 5.0)
