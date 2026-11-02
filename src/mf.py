"""轻量 Bias-MF（矩阵分解）对照模型：r̂(u,i) = μ + b_u + b_i + p_u·q_i。

定位：**扩展实验**，不是课程主线。它与 CF 使用完全相同的训练/测试划分、
相同的评分预测任务、相同的 MAE/RMSE 指标，因此可以放进同一张表比较；
但不参与推荐页演示，也不改变 E1–E4 的任何结果。

求解方式：交替最小二乘（ALS），每一轮依次对偏置与隐向量做岭回归闭式解。
相比块 SGD，ALS 不需要调学习率、在 10 万条评分上几秒即可收敛，
结果也更可复现（固定初始化 + 确定性迭代）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.data import TrainMatrix


@dataclass
class BiasMF:
    k: int = 20
    iterations: int = 15
    reg_factor: float = 10.0
    reg_bias: float = 5.0
    seed: int = 42

    def fit(self, train: TrainMatrix) -> "BiasMF":
        t0 = time.perf_counter()
        n_users, n_items = train.n_users, train.n_items
        rng = np.random.default_rng(self.seed)
        self.mu = float(train.global_mean)
        self.P = rng.normal(0, 0.01, size=(n_users, self.k))
        self.Q = rng.normal(0, 0.01, size=(n_items, self.k))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        R = train.R.tocsr()
        rows = [R.indices[R.indptr[u]:R.indptr[u + 1]] for u in range(n_users)]
        vals = [R.data[R.indptr[u]:R.indptr[u + 1]].astype(np.float64) for u in range(n_users)]
        # 物品侧的 (user, rating) 用转置矩阵准备
        RT = R.T.tocsr()
        icols = [RT.indices[RT.indptr[i]:RT.indptr[i + 1]] for i in range(n_items)]
        ivals = [RT.data[RT.indptr[i]:RT.indptr[i + 1]].astype(np.float64) for i in range(n_items)]

        eye = np.eye(self.k)
        for _ in range(self.iterations):
            # 1) 用户偏置
            for u in range(n_users):
                items = rows[u]
                if items.size == 0:
                    continue
                resid = vals[u] - self.mu - self.b_i[items] - self.Q[items] @ self.P[u]
                self.b_u[u] = resid.sum() / (items.size + self.reg_bias)
            # 2) 物品偏置
            for i in range(n_items):
                users = icols[i]
                if users.size == 0:
                    continue
                resid = ivals[i] - self.mu - self.b_u[users] - self.P[users] @ self.Q[i]
                self.b_i[i] = resid.sum() / (users.size + self.reg_bias)
            # 3) 用户隐向量
            for u in range(n_users):
                items = rows[u]
                if items.size == 0:
                    continue
                Qi = self.Q[items]
                target = vals[u] - self.mu - self.b_u[u] - self.b_i[items]
                A = Qi.T @ Qi + self.reg_factor * eye
                self.P[u] = np.linalg.solve(A, Qi.T @ target)
            # 4) 物品隐向量
            for i in range(n_items):
                users = icols[i]
                if users.size == 0:
                    continue
                Pu = self.P[users]
                target = ivals[i] - self.mu - self.b_u[users] - self.b_i[i]
                A = Pu.T @ Pu + self.reg_factor * eye
                self.Q[i] = np.linalg.solve(A, Pu.T @ target)

        self.train_seconds = time.perf_counter() - t0
        return self

    def predict_pairs(self, pairs: np.ndarray) -> np.ndarray:
        u = pairs[:, 0]
        i = pairs[:, 1]
        pred = (self.mu + self.b_u[u] + self.b_i[i]
                + np.einsum("ij,ij->i", self.P[u], self.Q[i]))
        return np.clip(pred, 1.0, 5.0)

    def score_all(self, u_idx: int) -> np.ndarray:
        """一次性给该用户算全部电影的预测分（供 Top-N 使用，本实验未使用）。"""
        return np.clip(self.mu + self.b_u[u_idx] + self.b_i + self.Q @ self.P[u_idx], 1.0, 5.0)
