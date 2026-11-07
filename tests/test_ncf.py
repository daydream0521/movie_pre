"""E5 的最小测试。

在 deeplearning 环境（有 torch）里直接运行：
    python tests/test_ncf.py -v
在基础环境（无 torch）里 pytest 会自动跳过，不影响原有测试。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch

    from src.ncf import NCFMLP, TrainConfig, predict, train_fold
    HAS_TORCH = True
except Exception:                                    # pragma: no cover - 基础环境走这里
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch（conda 环境 deeplearning）")
class TestNCF(unittest.TestCase):
    def test_forward_shape_and_prediction_range(self) -> None:
        cfg = TrainConfig(device="cpu", max_epochs=1, batch_size=64, emb_dim=8, hidden=(16, 8))
        model = NCFMLP(10, 7, cfg)
        out = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
        self.assertEqual(tuple(out.shape), (3,))
        preds = predict(model, np.array([0, 1, 2]), np.array([3, 4, 5]))
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue(((preds >= 1.0) & (preds <= 5.0)).all())

    def test_early_stopping_stops_early_and_keeps_best(self) -> None:
        # 常数标签：模型几轮内就拟合完，验证损失不再下降 → 必须提前停下
        n = 300
        u = np.arange(n)
        i = np.arange(n)
        r = np.full(n, 3.0, dtype=np.float32)
        cfg = TrainConfig(device="cpu", max_epochs=40, patience=2, batch_size=64,
                          emb_dim=8, hidden=(16, 8), lr=5e-3)
        res = train_fold(u, i, r, n, n, cfg, seed_offset=0)
        self.assertLess(res["epochs_run"], 40, "早停没有生效")
        self.assertGreaterEqual(res["best_epoch"], 1)
        self.assertLessEqual(res["best_epoch"], res["epochs_run"])
        self.assertEqual(res["n_train"] + res["n_val"], n)
        self.assertEqual(res["n_val"], int(round(n * cfg.val_ratio)))
        self.assertLess(res["best_val_rmse"], res["history"][0]["val_rmse"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
