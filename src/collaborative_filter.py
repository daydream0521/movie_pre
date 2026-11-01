"""User-CF 与 Item-CF 的核心实现（自行实现，不调用黑盒 predict）。

统一定义（代码、界面、报告、讲稿共用这一套）：

- 相似度只用训练集评分计算。
- 用户向量：本用户评分减去**自身训练均值**，只在观测位置非零（中心化余弦）。
- 物品向量：评分减去**该条评分的用户均值**（消除不同用户打分尺度差异），再做余弦（调整余弦）。
- 相似度收缩：sim * n/(n+10)，n 为两用户/两物品的共同评分数；n < min_common(默认 3) 时置 0。
- 相似度矩阵对角线置 0（自己不能当自己的邻居）。
- User-CF 预测：pred(u,i) = mean(u) + Σ_{v∈N_K} sim(u,v)·(r(v,i)-mean(v)) / Σ|sim(u,v)|
  其中 N_K 是“对 i 有过评分、且相似度为正”的邻居里相似度最高的至多 K 个。
- Item-CF 预测：pred(u,i) = mean(u) + Σ_{j∈N_K} sim(i,j)·(r(u,j)-mean(u)) / Σ|sim(i,j)|
  其中 N_K 是“目标用户在训练集评过、且与 i 相似度为正”的物品里相似度最高的至多 K 个。
- 无有效邻居 → 回退到该用户训练均值；用户训练集为空 → 回退全局均值。预测裁剪到 [1,5]。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import TrainMatrix

ALGO_USER = "user"
ALGO_ITEM = "item"
ALGO_BASELINE = "baseline"


@dataclass
class PredictionResult:
    """批量预测结果，附带解释所需的中间量统计。"""

    preds: np.ndarray
    n_neighbors: np.ndarray
    is_fallback: np.ndarray

    @property
    def coverage(self) -> float:
        return float(1.0 - self.is_fallback.mean()) if len(self.preds) else 0.0


@dataclass
class Explanation:
    """单个 (用户, 电影) 预测的可解释拆解。"""

    user_id: int
    item_id: int
    algo: str
    base: float                      # mean(u)
    pred: float
    sim_sum: float
    is_fallback: bool
    neighbors: pd.DataFrame = field(default_factory=pd.DataFrame)
    formula: str = ""


class CFModel:
    """在给定训练矩阵上计算相似度并做预测 / 推荐。"""

    def __init__(
        self,
        train: TrainMatrix,
        min_common: int = 3,
        shrink: float = 10.0,
    ) -> None:
        self.train = train
        self.min_common = int(min_common)
        self.shrink = float(shrink)

        self.X = np.asarray(train.R.todense(), dtype=np.float32)
        self.M = np.asarray(train.M.todense(), dtype=bool)
        self.global_mean = float(train.global_mean)
        self.mu = train.user_means()
        # 中心化矩阵：Xc[u,i] = r(u,i) - mean(u)，只在观测位置非零
        self.Xc = np.where(self.M, self.X - self.mu[:, None], 0.0).astype(np.float32)

        self.S_user: np.ndarray | None = None
        self.S_item: np.ndarray | None = None
        self.timing: dict[str, float] = {}

    @property
    def n_users(self) -> int:
        return self.train.n_users

    @property
    def n_items(self) -> int:
        return self.train.n_items

    # ------------------------------------------------------------------ 相似度
    def build_user_similarity(self) -> np.ndarray:
        t0 = time.perf_counter()
        norms = np.sqrt((self.Xc ** 2).sum(axis=1))
        Xn = self.Xc / np.maximum(norms, 1e-12)[:, None]
        Xn[norms <= 1e-12] = 0.0
        sim = Xn @ Xn.T
        co = self.M.astype(np.float32) @ self.M.astype(np.float32).T
        sim = self._postprocess(sim, co)
        self.S_user = sim
        self.timing["build_user_similarity"] = time.perf_counter() - t0
        return sim

    def build_item_similarity(self) -> np.ndarray:
        t0 = time.perf_counter()
        norms = np.sqrt((self.Xc ** 2).sum(axis=0))          # 每个物品的中心化向量范数
        Xn = self.Xc / np.maximum(norms, 1e-12)[None, :]
        Xn[:, norms <= 1e-12] = 0.0
        sim = Xn.T @ Xn                                       # 调整余弦（已减去用户均值）
        co = self.M.astype(np.float32).T @ self.M.astype(np.float32)
        sim = self._postprocess(sim, co)
        self.S_item = sim
        self.timing["build_item_similarity"] = time.perf_counter() - t0
        return sim

    def _postprocess(self, sim: np.ndarray, co: np.ndarray) -> np.ndarray:
        # 共同评分数越少，相似度越不可信，按 n/(n+shrink) 收缩；co=0 处保持 0
        with np.errstate(invalid="ignore", divide="ignore"):
            shrink = np.where(co > 0, co / np.maximum(co + self.shrink, 1e-12), 0.0)
        sim = (sim * shrink).astype(np.float32)
        sim[co < self.min_common] = 0.0           # 证据不足的相似对直接不参与
        np.fill_diagonal(sim, 0.0)
        return sim

    def ensure_similarity(self, algo: str) -> np.ndarray:
        if algo == ALGO_USER:
            if self.S_user is None:
                self.build_user_similarity()
            return self.S_user
        if algo == ALGO_ITEM:
            if self.S_item is None:
                self.build_item_similarity()
            return self.S_item
        raise ValueError(f"未知算法: {algo}")

    # ------------------------------------------------------------------ 预测
    def predict(self, pairs: np.ndarray, algo: str, K: int = 30) -> PredictionResult:
        """批量预测。pairs 形如 [(user_idx, item_idx), ...]，全部使用训练矩阵。"""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        sim = self.ensure_similarity(algo)
        preds = np.zeros(len(pairs), dtype=np.float64)
        n_nbr = np.zeros(len(pairs), dtype=np.int32)
        fallback = np.ones(len(pairs), dtype=bool)

        order = np.argsort(pairs[:, 0], kind="stable")
        users_sorted = pairs[order, 0]
        boundaries = np.flatnonzero(np.diff(users_sorted)) + 1
        for group in np.split(order, boundaries):
            if len(group) == 0:
                continue
            u = int(pairs[group[0], 0])
            items = pairs[group, 1]
            if algo == ALGO_USER:
                p, k, fb = self._predict_user_group(u, items, sim, K)
            else:
                p, k, fb = self._predict_item_group(u, items, sim, K)
            preds[group], n_nbr[group], fallback[group] = p, k, fb
        return PredictionResult(preds=np.clip(preds, 1.0, 5.0), n_neighbors=n_nbr, is_fallback=fallback)

    def predict_baseline(self, pairs: np.ndarray) -> PredictionResult:
        """用户均值基准：pred = mean(u)，无评分用户回退全局均值。"""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        users = pairs[:, 0]
        has = self.M.sum(axis=1) > 0
        preds = np.where(has[users], self.mu[users], self.global_mean)
        fallback = ~has[users]
        return PredictionResult(
            preds=np.clip(preds, 1.0, 5.0), n_neighbors=np.zeros(len(pairs), dtype=np.int32),
            is_fallback=fallback,
        )

    def _user_neighbor_order(self, u: int, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按相似度降序返回正相似邻居索引与权重。"""
        row = sim[u]
        nbrs = np.flatnonzero(row > 0)
        if nbrs.size == 0:
            return nbrs, np.empty(0, dtype=np.float32)
        nbrs = nbrs[np.argsort(-row[nbrs], kind="stable")]
        return nbrs, row[nbrs]

    def _predict_user_group(self, u: int, items: np.ndarray, sim: np.ndarray, K: int):
        nbrs, w = self._user_neighbor_order(u, sim)
        if nbrs.size == 0:
            base = self.mu[u] if self.M[u].any() else self.global_mean
            return np.full(len(items), base), np.zeros(len(items), np.int32), np.ones(len(items), bool)

        sub = self.X[np.ix_(nbrs, items)]                    # (邻居数, 目标电影数)
        valid = sub > 0
        # 每个物品只保留“按相似度降序的前 K 个已评分邻居”
        keep = valid & (np.cumsum(valid, axis=0) <= K)
        weighted = keep * w[:, None]
        num = (weighted * (sub - self.mu[nbrs][:, None])).sum(axis=0)
        den = weighted.sum(axis=0)
        ok = den > 0
        base = self.mu[u] if self.M[u].any() else self.global_mean
        preds = np.full(len(items), base, dtype=np.float64)
        preds[ok] = base + num[ok] / den[ok]
        return preds, keep.sum(axis=0).astype(np.int32), ~ok

    def _predict_item_group(self, u: int, items: np.ndarray, sim: np.ndarray, K: int):
        cand = np.flatnonzero(self.M[u])                      # 该用户训练集评过的物品
        base = self.mu[u] if cand.size else self.global_mean
        if cand.size == 0:
            return np.full(len(items), base), np.zeros(len(items), np.int32), np.ones(len(items), bool)

        sub = sim[np.ix_(items, cand)]                        # (目标电影数, 候选相似电影数)，对角线已置 0
        delta = (self.X[u, cand] - base).astype(np.float32)   # 目标用户对相似电影的“超均值”部分
        k = min(K, cand.size)
        order = np.argsort(-sub, axis=1, kind="stable")[:, :k]
        sim_top = np.take_along_axis(sub, order, axis=1)
        valid = sim_top > 0
        delta_top = np.take_along_axis(np.broadcast_to(delta, sub.shape), order, axis=1)
        weighted = sim_top * valid
        num = (weighted * delta_top).sum(axis=1)
        den = weighted.sum(axis=1)
        ok = den > 0
        preds = np.full(len(items), base, dtype=np.float64)
        preds[ok] = base + num[ok] / den[ok]
        return preds, valid.sum(axis=1).astype(np.int32), ~ok

    # ------------------------------------------------------------------ 解释
    def explain(self, user_idx: int, item_idx: int, algo: str, K: int = 30) -> Explanation:
        """拆解单个预测：邻居、相似度、原始评分、各自贡献。"""
        sim = self.ensure_similarity(algo)
        base = self.mu[user_idx] if self.M[user_idx].any() else self.global_mean

        if algo == ALGO_USER:
            nbrs, w = self._user_neighbor_order(user_idx, sim)
            rated = nbrs[self.X[nbrs, item_idx] > 0]
            rated = rated[np.argsort(-sim[user_idx, rated], kind="stable")][:K]
            if rated.size == 0:
                return Explanation(
                    user_id=int(self.train.user_ids[user_idx]), item_id=int(self.train.item_ids[item_idx]),
                    algo=algo, base=base, pred=base, sim_sum=0.0, is_fallback=True,
                    neighbors=pd.DataFrame(
                        columns=["neighbor_user_id", "similarity", "neighbor_mean", "rating", "contribution"]
                    ),
                    formula="无可用的正相似邻居 → 回退用户均值",
                )
            sims = sim[user_idx, rated]
            ratings = self.X[rated, item_idx]
            contrib = sims * (ratings - self.mu[rated])
            den = np.abs(sims).sum()
            table = pd.DataFrame({
                "neighbor_user_id": self.train.user_ids[rated].astype(int),
                "similarity": sims,
                "neighbor_mean": self.mu[rated],
                "rating": ratings,
                "contribution": contrib / den,
            }).sort_values("contribution", ascending=False)
            return Explanation(
                user_id=int(self.train.user_ids[user_idx]), item_id=int(self.train.item_ids[item_idx]),
                algo=algo, base=base, pred=float(np.clip(base + contrib.sum() / den, 1, 5)),
                sim_sum=float(den), is_fallback=False, neighbors=table,
                formula="pred = mean(u) + Σ sim(u,v)·(r(v,i)-mean(v)) / Σ|sim(u,v)|",
            )

        cand = np.flatnonzero(self.M[user_idx])
        if cand.size == 0:
            return Explanation(
                user_id=int(self.train.user_ids[user_idx]), item_id=int(self.train.item_ids[item_idx]),
                algo=algo, base=base, pred=base, sim_sum=0.0, is_fallback=True,
                neighbors=pd.DataFrame(
                    columns=["neighbor_item_id", "similarity", "user_rating", "centered_rating", "contribution"]
                ),
                formula="该用户在训练集中没有评分 → 回退全局均值",
            )
        sims = sim[item_idx, cand]
        pos = sims > 0
        cand, sims = cand[pos], sims[pos]
        top = np.argsort(-sims, kind="stable")[:K]
        cand, sims = cand[top], sims[top]
        if cand.size == 0:
            return Explanation(
                user_id=int(self.train.user_ids[user_idx]), item_id=int(self.train.item_ids[item_idx]),
                algo=algo, base=base, pred=base, sim_sum=0.0, is_fallback=True,
                neighbors=pd.DataFrame(
                    columns=["neighbor_item_id", "similarity", "user_rating", "centered_rating", "contribution"]
                ),
                formula="无可用的正相似物品 → 回退用户均值",
            )
        ratings = self.X[user_idx, cand]
        centered = ratings - base
        den = np.abs(sims).sum()
        table = pd.DataFrame({
            "neighbor_item_id": self.train.item_ids[cand].astype(int),
            "similarity": sims,
            "user_rating": ratings,
            "centered_rating": centered,
            "contribution": sims * centered / den,
        }).sort_values("contribution", ascending=False)
        return Explanation(
            user_id=int(self.train.user_ids[user_idx]), item_id=int(self.train.item_ids[item_idx]),
            algo=algo, base=base, pred=float(np.clip(base + sims.dot(centered) / den, 1, 5)),
            sim_sum=float(den), is_fallback=False, neighbors=table,
            formula="pred = mean(u) + Σ sim(i,j)·(r(u,j)-mean(u)) / Σ|sim(i,j)|",
        )

    # ------------------------------------------------------------------ 推荐
    def recommend(
        self, user_idx: int, algo: str, K: int = 30, n: int = 10, candidate_idx: np.ndarray | None = None
    ) -> pd.DataFrame:
        """对全目录（默认排除训练集已评分电影）生成 Top-N，并标注是否走过回退。"""
        sim = self.ensure_similarity(algo)
        if candidate_idx is None:
            candidate_idx = np.flatnonzero(~self.M[user_idx])
        else:
            candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
        if candidate_idx.size == 0:
            return pd.DataFrame(columns=["item_idx", "score", "n_neighbors", "is_fallback"])

        if algo == ALGO_USER:
            preds, k, fb = self._predict_user_group(user_idx, candidate_idx, sim, K)
        else:
            preds, k, fb = self._predict_item_group(user_idx, candidate_idx, sim, K)

        df = pd.DataFrame({
            "item_idx": candidate_idx,
            "score": np.clip(preds, 1.0, 5.0),
            "n_neighbors": k,
            "is_fallback": fb,
        })
        # 有真实邻居支撑的推荐优先；同分时邻居越多越靠前
        df = df.sort_values(["is_fallback", "score", "n_neighbors", "item_idx"], ascending=[True, False, False, True])
        return df.head(n).reset_index(drop=True)

    def neighbor_stats(self, user_idx: int, algo: str, item_idx: int | None = None) -> dict:
        """界面用诊断信息：正相似邻居总数、目标电影上可用邻居数。"""
        sim = self.ensure_similarity(algo)
        if algo == ALGO_USER:
            row = sim[user_idx]
            positive = int((row > 0).sum())
            avail = int((row[self.X[:, item_idx] > 0] > 0).sum()) if item_idx is not None else None
            return {"positive_neighbors": positive, "available_for_item": avail}
        cand = np.flatnonzero(self.M[user_idx])
        positive = int((sim[:, cand] > 0).any(axis=0).sum()) if cand.size else 0
        avail = int((sim[item_idx, cand] > 0).sum()) if (item_idx is not None and cand.size) else None
        return {"positive_neighbors": positive, "available_for_item": avail}
