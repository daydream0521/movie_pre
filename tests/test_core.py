"""只覆盖“最容易出错”的关键点：缺失值、公式一致性、已看过滤、训练/测试隔离、K 语义。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.collaborative_filter import ALGO_ITEM, ALGO_USER, CFModel  # noqa: E402
from src.data import TrainMatrix, build_train_matrix  # noqa: E402
from src.tutorial import TUTORIAL_RATINGS, build_tutorial_model, hand_calculation  # noqa: E402

DATA_OK = (ROOT / "data" / "ml-100k" / "u.data").exists()


def tiny_matrix(rows: list[list[float | None]]) -> TrainMatrix:
    r_idx, c_idx, vals = [], [], []
    for u, row in enumerate(rows):
        for i, v in enumerate(row):
            if v is not None:
                r_idx.append(u)
                c_idx.append(i)
                vals.append(float(v))
    shape = (len(rows), len(rows[0]))
    R = sparse.csr_matrix((vals, (r_idx, c_idx)), shape=shape, dtype=np.float32)
    M = sparse.csr_matrix((np.ones(len(vals), dtype=np.float32), (r_idx, c_idx)), shape=shape)
    return TrainMatrix(
        R=R, M=M, user_ids=np.arange(1, shape[0] + 1),
        item_ids=np.arange(1, shape[1] + 1),
        global_mean=float(np.mean(vals)),
    )


def test_missing_is_not_zero():
    """缺失评分不能拉低用户均值。"""
    train = tiny_matrix([[5, None, 1], [4, 4, 4]])
    means = train.user_means()
    assert means[0] == pytest.approx(3.0)      # (5+1)/2，而不是 (5+0+1)/3
    assert means[1] == pytest.approx(4.0)
    assert train.n_ratings == 5
    assert train.density == pytest.approx(5 / 6)


def test_user_cf_prediction_matches_formula():
    """手算核对：pred = mean(u) + Σ sim·(r-mean) / Σ|sim|。"""
    model, _ = build_tutorial_model()          # min_common=2, 无收缩
    ex = model.explain(0, 3, ALGO_USER, K=3)
    manual = ex.base + sum(
        r["similarity"] * (r["rating"] - r["neighbor_mean"]) for _, r in ex.neighbors.iterrows()
    ) / ex.sim_sum
    assert not ex.is_fallback
    assert ex.pred == pytest.approx(min(max(manual, 1.0), 5.0), abs=1e-4)


def test_item_cf_prediction_matches_formula():
    model, _ = build_tutorial_model()
    ex = model.explain(0, 3, ALGO_ITEM, K=3)
    manual = ex.base + sum(
        r["similarity"] * (r["user_rating"] - ex.base) for _, r in ex.neighbors.iterrows()
    ) / ex.sim_sum
    assert not ex.is_fallback
    assert ex.pred == pytest.approx(min(max(manual, 1.0), 5.0), abs=1e-4)


def test_k_limits_actual_neighbors():
    """K=1 时只用一个邻居，预测必然等于 base + 该邻居偏差。"""
    model, _ = build_tutorial_model()
    ex1 = model.explain(0, 3, ALGO_USER, K=1)
    top = ex1.neighbors.iloc[0]
    assert len(ex1.neighbors) == 1
    assert ex1.pred == pytest.approx(
        np.clip(ex1.base + (top["rating"] - top["neighbor_mean"]), 1, 5), abs=1e-4
    )
    ex3 = model.explain(0, 3, ALGO_USER, K=3)
    assert len(ex3.neighbors) >= 2
    assert ex1.pred != pytest.approx(ex3.pred, abs=1e-6)


def test_no_neighbor_falls_back_to_user_mean():
    """完全孤立的目标用户 → 均值回退，且被显式标记。"""
    model, _ = build_tutorial_model()
    train = tiny_matrix([[5, None], [None, None]])
    m2 = CFModel(train, min_common=1, shrink=0.0)
    for algo in (ALGO_USER, ALGO_ITEM):
        res = m2.predict(np.array([[1, 0]]), algo, K=5)
        assert bool(res.is_fallback[0])
        assert res.preds[0] == pytest.approx(m2.global_mean, abs=1e-6)


def test_negative_similarity_not_used():
    """口味相反的用户不能把预测往反方向拉。"""
    model, _ = build_tutorial_model()
    sim = model.ensure_similarity(ALGO_USER)
    assert sim[0, 3] < 0                      # Alice vs Dave 负相似
    ex = model.explain(0, 3, ALGO_USER, K=4)
    assert all(r["similarity"] > 0 for _, r in ex.neighbors.iterrows())


@pytest.mark.skipif(not DATA_OK, reason="需要先下载 MovieLens 100K")
def test_recommend_excludes_seen_items():
    from src.data import build_train_matrix, load_catalog, load_fold

    _, _, users, items = load_catalog(ROOT / "data" / "ml-100k")
    train_df, _ = load_fold(ROOT / "data" / "ml-100k", 1)
    model = CFModel(build_train_matrix(train_df, users, items))
    for algo in (ALGO_USER, ALGO_ITEM):
        rec = model.recommend(0, algo, K=30, n=20)
        seen = set(np.flatnonzero(model.M[0]))
        assert not (set(rec["item_idx"]) & seen)
        assert len(rec) == 20


@pytest.mark.skipif(not DATA_OK, reason="需要先下载 MovieLens 100K")
def test_test_ratings_never_in_training_matrix():
    from src.data import build_train_matrix, load_catalog, load_fold

    _, _, users, items = load_catalog(ROOT / "data" / "ml-100k")
    train_df, test_df = load_fold(ROOT / "data" / "ml-100k", 1)
    model = CFModel(build_train_matrix(train_df, users, items))
    uidx = {int(u): i for i, u in enumerate(users)}
    iidx = {int(m): i for i, m in enumerate(items)}
    # 官方 fold 的 test 与 base 是互斥的
    overlap = sum(1 for _, r in test_df.iterrows() if model.M[uidx[r.user_id], iidx[r.item_id]])
    assert overlap == 0
    assert model.train.n_ratings == len(train_df) == 80000


def test_hand_calculation_is_reproducible():
    txt = hand_calculation(ALGO_USER)
    assert "预测 =" in txt and "均值" in txt
    assert "0.704" in txt or "0.546" in txt   # 真实算出的相似度出现在推导里
    assert len(hand_calculation(ALGO_ITEM).splitlines()) <= 12


def test_tutorial_has_missing_cells():
    assert any(v is None for row in TUTORIAL_RATINGS for v in row)
