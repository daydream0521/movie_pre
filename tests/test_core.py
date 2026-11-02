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


# --------------------------------------------------------------- 教学页一致性
def test_step3_neighbors_equal_step4_neighbors():
    """第 3 步选出的邻居必须与第 4 步实际参与加权的人完全一致。

    曾出现的 bug：第 3 步只按相似度 >0 取 Top-K，没要求「给目标电影评过分」，
    于是表中会出现第 4 步根本没用到的人。
    """
    from src.tutorial import TUTORIAL_MOVIES, TUTORIAL_RATINGS, TUTORIAL_USERS, movie_label, steps

    id2idx = {m[0]: k for k, m in enumerate(TUTORIAL_MOVIES)}
    for algo in (ALGO_USER, ALGO_ITEM):
        for u in range(len(TUTORIAL_USERS)):
            for i in range(len(TUTORIAL_MOVIES)):
                hide = (u, i) if TUTORIAL_RATINGS[u][i] is not None else None
                for K in (1, 2, 3, 4):
                    st = steps(algo, u, i, K, hide=hide)
                    table = st[2]["data"]["table"]
                    model, _ = build_tutorial_model(hide=hide)
                    ex = model.explain(u, i, algo, K=K)
                    if algo == ALGO_USER:
                        got = set(table["用户"])
                        want = (set() if ex.is_fallback else
                                {TUTORIAL_USERS[int(x) - 1] for x in ex.neighbors["neighbor_user_id"]})
                    else:
                        got = set(table["候选电影"])
                        want = (set() if ex.is_fallback else
                                {movie_label(id2idx[int(x)]) for x in ex.neighbors["neighbor_item_id"]})
                    assert got == want, f"{algo} user={u} item={i} K={K}: {got} != {want}"


def test_rated_target_is_hidden_from_matrix():
    """目标用户已评分时，该条评分必须从建模矩阵里移除，否则「预测」没有意义。"""
    from src.tutorial import TUTORIAL_USERS, steps, target_is_rated, true_rating

    assert target_is_rated(0, 0) and true_rating(0, 0) == 5.0
    model, train = build_tutorial_model(hide=(0, 0))
    assert train.M[0, 0] == 0
    assert train.n_ratings == sum(1 for row in TUTORIAL_RATINGS for v in row if v is not None) - 1
    st = steps(ALGO_USER, 0, 0, 3, hide=(0, 0))
    assert st[0]["data"]["hidden"] is True
    assert "隐藏" in st[-1]["body"]                 # 结论里要交代真实评分
    # 未隐藏时，同一格仍然在矩阵里
    _, train_full = build_tutorial_model()
    assert train_full.M[0, 0] == 1


def test_baseline_has_no_coverage():
    """用户均值基准不需要邻居，覆盖率必须是不适用（NaN），不能算成 100%。"""
    model, _ = build_tutorial_model()
    res = model.predict_baseline(np.array([[0, 3], [1, 0]]))
    assert np.isnan(res.coverage)
    assert not res.coverage_applicable
    cf = model.predict(np.array([[0, 3]]), ALGO_USER, K=3)
    assert not np.isnan(cf.coverage)


def test_recommend_ranks_by_weighted_sum():
    """Top-N 用加权和排序：非回退条目按 rank_score 严格不增，分数与 explain 的分子一致。"""
    model, _ = build_tutorial_model()
    rec = model.recommend(0, ALGO_USER, K=3, n=4)
    assert not rec.empty
    live = rec[~rec["is_fallback"]]
    assert list(live["rank_score"]) == sorted(live["rank_score"], reverse=True)
    top = rec.iloc[0]
    ex = model.explain(0, int(top["item_idx"]), ALGO_USER, K=3)
    assert top["rank_score"] == pytest.approx(ex.rank_score, abs=1e-5)
    # 预测分仍然由 mean + 分子/分母 得到
    assert top["pred"] == pytest.approx(np.clip(ex.base + ex.rank_score / ex.sim_sum, 1, 5), abs=1e-4)


def test_ndcg_and_topk_helpers():
    from src.evaluation import ndcg_at_k, topk_from_scores

    rel = {1, 2}
    assert ndcg_at_k(np.array([1, 2, 3]), rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([9, 9, 9]), rel, 3) == 0.0
    # 命中排在第 1 位应高于命中排在第 3 位
    assert ndcg_at_k(np.array([1, 9, 9]), rel, 3) > ndcg_at_k(np.array([9, 9, 1]), rel, 3)

    scores = np.array([0.1, 0.9, 0.5, 0.7])
    seen = np.array([False, True, False, False])   # 已看过 1 号，不该出现
    top = topk_from_scores(scores, seen, 2)
    assert list(top) == [3, 2]


def test_bias_mf_fits_and_predicts_in_range():
    """扩展模型的最小冒烟：ALS 收敛后预测必须在 [1,5]，且不产生 NaN。"""
    from src.mf import BiasMF

    _, train = build_tutorial_model()
    mf = BiasMF(k=5, iterations=10).fit(train)
    pairs = np.array([[0, 3], [1, 4], [4, 0]])
    preds = mf.predict_pairs(pairs)
    assert np.isfinite(preds).all()
    assert ((preds >= 1.0) & (preds <= 5.0)).all()
    assert mf.score_all(0).shape == (train.n_items,)
