"""实验流程：E1 五折比较、E2 邻居数 K、E3 训练记录稀疏化，以及指标与作图。

所有指标都由真实预测与测试集真实评分比对产生，直接写入 results/*.csv。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.collaborative_filter import ALGO_BASELINE, ALGO_ITEM, ALGO_USER, CFModel
from src.data import build_train_matrix, load_fold

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


# --------------------------------------------------------------------- 指标
def pair_metrics(preds: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    err = np.asarray(preds, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def _index_pairs(test: pd.DataFrame, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    uidx = {int(u): i for i, u in enumerate(users)}
    iidx = {int(m): i for i, m in enumerate(items)}
    return np.column_stack([
        test["user_id"].map(uidx).to_numpy(),
        test["item_id"].map(iidx).to_numpy(),
    ]).astype(np.int64)


# --------------------------------------------------------------------- 单折
def evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    users: np.ndarray,
    items: np.ndarray,
    algos: tuple[str, ...] = (ALGO_BASELINE, ALGO_USER, ALGO_ITEM),
    K: int = 30,
) -> list[dict]:
    t0 = time.perf_counter()
    train = build_train_matrix(train_df, users, items)
    pairs = _index_pairs(test_df, users, items)
    actual = test_df["rating"].to_numpy(dtype=np.float64)
    model = CFModel(train)

    rows: list[dict] = []
    for algo in algos:
        build_time = 0.0
        if algo == ALGO_USER:
            t = time.perf_counter()
            model.ensure_similarity(ALGO_USER)
            build_time = time.perf_counter() - t
        elif algo == ALGO_ITEM:
            t = time.perf_counter()
            model.ensure_similarity(ALGO_ITEM)
            build_time = time.perf_counter() - t

        t = time.perf_counter()
        if algo == ALGO_BASELINE:
            res = model.predict_baseline(pairs)
        else:
            res = model.predict(pairs, algo, K=K)
        predict_time = time.perf_counter() - t

        mae, rmse = pair_metrics(res.preds, actual)
        rows.append({
            "algorithm": algo,
            "K": K if algo != ALGO_BASELINE else "-",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            # 覆盖率只对 CF 定义：基准模型不需要邻居 → NaN（CSV 中为空），不参与比较
            "coverage": round(res.coverage, 4),
            "coverage_defined": int(res.coverage_applicable),
            "n_pairs": int(len(pairs)),
            "fallback_count": int(res.is_fallback.sum()),
            "fallback_rate": round(float(res.is_fallback.mean()), 4),
            "mean_neighbors": round(float(res.n_neighbors.mean()), 2),
            "build_seconds": round(build_time, 3),
            "predict_seconds": round(predict_time, 3),
        })
    for r in rows:
        r["total_seconds"] = round(time.perf_counter() - t0, 3)
    return rows


# --------------------------------------------------------------------- E1
def experiment_e1(data_dir: str | Path, users: np.ndarray, items: np.ndarray, K: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """五个官方 fold × 三个模型。"""
    rows = []
    for fold in range(1, 6):
        train_df, test_df = load_fold(data_dir, fold)
        fold_rows = evaluate_fold(train_df, test_df, users, items, K=K)
        for r in fold_rows:
            r["fold"] = fold
            r["train_size"] = len(train_df)
            r["test_size"] = len(test_df)
        rows.extend(fold_rows)
        print(f"  fold {fold} 完成：" + ", ".join(
            f"{r['algorithm']} RMSE={r['rmse']:.4f}" for r in fold_rows
        ))
    df = pd.DataFrame(rows)

    summary = (
        df.groupby("algorithm")
        .agg(
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
            fallback_rate_mean=("fallback_rate", "mean"),
            build_seconds_mean=("build_seconds", "mean"),
            predict_seconds_mean=("predict_seconds", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return df, summary


# --------------------------------------------------------------------- E2
def experiment_e2(
    data_dir: str | Path, users: np.ndarray, items: np.ndarray,
    Ks: tuple[int, ...] = (5, 10, 20, 30, 50), fold: int = 1,
) -> pd.DataFrame:
    train_df, test_df = load_fold(data_dir, fold)
    train = build_train_matrix(train_df, users, items)
    pairs = _index_pairs(test_df, users, items)
    actual = test_df["rating"].to_numpy(dtype=np.float64)
    model = CFModel(train)
    model.ensure_similarity(ALGO_USER)
    model.ensure_similarity(ALGO_ITEM)

    rows = []
    for K in Ks:
        for algo in (ALGO_USER, ALGO_ITEM):
            t = time.perf_counter()
            res = model.predict(pairs, algo, K=K)
            el = time.perf_counter() - t
            mae, rmse = pair_metrics(res.preds, actual)
            rows.append({
                "fold": fold, "algorithm": algo, "K": K,
                "mae": round(mae, 4), "rmse": round(rmse, 4),
                "coverage": round(res.coverage, 4),
                "fallback_count": int(res.is_fallback.sum()),
                "mean_neighbors": round(float(res.n_neighbors.mean()), 2),
                "predict_seconds": round(el, 3),
            })
        print(f"  K={K} 完成")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E3
def subsample_train(train_df: pd.DataFrame, ratio: float, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """按用户随机保留指定比例的训练记录，每个用户至少保留 1 条。"""
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    bumped = 0
    for _, g in train_df.groupby("user_id", sort=True):
        idx = g.index.to_numpy()
        k = max(1, int(round(len(idx) * ratio)))
        if ratio < 1.0 and k > len(idx) * ratio:
            bumped += 1
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    out = train_df.loc[np.concatenate(keep)].sort_values(["user_id", "item_id"]).reset_index(drop=True)
    info = {
        "target_ratio": ratio,
        "actual_ratio": round(len(out) / len(train_df), 4),
        "rows": len(out),
        "users_kept": out["user_id"].nunique(),
        "users_with_single_rating": int((out.groupby("user_id").size() == 1).sum()),
        "bumped_to_one": bumped,
        "seed": seed,
    }
    return out, info


def experiment_e3(
    data_dir: str | Path, users: np.ndarray, items: np.ndarray,
    ratios: tuple[float, ...] = (1.0, 0.5, 0.25), K: int = 30, fold: int = 1, seed: int = 42,
) -> pd.DataFrame:
    base_train, test_df = load_fold(data_dir, fold)
    pairs = _index_pairs(test_df, users, items)
    actual = test_df["rating"].to_numpy(dtype=np.float64)
    rows = []
    for ratio in ratios:
        sub, info = subsample_train(base_train, ratio, seed=seed)
        train = build_train_matrix(sub, users, items)
        model = CFModel(train)
        model.ensure_similarity(ALGO_USER)
        model.ensure_similarity(ALGO_ITEM)
        for algo in (ALGO_USER, ALGO_ITEM):
            res = model.predict(pairs, algo, K=K)
            mae, rmse = pair_metrics(res.preds, actual)
            rows.append({
                "fold": fold, "target_ratio": ratio, "actual_ratio": info["actual_ratio"],
                "train_rows": info["rows"], "algorithm": algo, "K": K,
                "mae": round(mae, 4), "rmse": round(rmse, 4),
                "coverage": round(res.coverage, 4),
                "fallback_count": int(res.is_fallback.sum()),
                "density_pct": round(100 * info["rows"] / (len(users) * len(items)), 3),
                "users_single_rating": info["users_with_single_rating"],
            })
        print(f"  保留比例 {ratio:.0%}（实际 {info['actual_ratio']:.1%}，{info['rows']} 条）完成")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E4
ALGO_POPULAR = "popular"


def popularity_scores(model: CFModel) -> np.ndarray:
    """热门度 = 该电影在训练集中被评分的次数（不使用任何测试信息）。"""
    return np.asarray(model.M.sum(axis=0)).ravel().astype(np.float64)


def topk_from_scores(scores: np.ndarray, seen_mask: np.ndarray, k: int) -> np.ndarray:
    """在「训练集未评分」的候选里取分数最高的 k 个；同分按 item_idx 升序，保证可复现。"""
    cand = np.flatnonzero(~seen_mask)
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((cand, -scores[cand]))
    return cand[order[:k]]


def ndcg_at_k(rec: np.ndarray, relevant: set[int], k: int) -> float:
    """二值相关性的 NDCG@k：相关记 1，不相关记 0。"""
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(rec[:k]) if int(item) in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def experiment_e4(
    data_dir: str | Path,
    users: np.ndarray,
    items: np.ndarray,
    K: int = 30,
    n: int = 10,
    folds: tuple[int, ...] = (1, 2, 3, 4, 5),
    relevant_threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top-N 推荐评测：热门推荐 / User-CF / Item-CF。

    设定（与 E1 的评分预测任务不同，报告里必须分开表述）：
    - 相关定义：测试集中评分 ≥ 4 的电影；
    - 候选集：全部未在训练集评分的电影（三个模型完全一致）；
    - 指标：Recall@N、NDCG@N、HitRate@N，按用户计算后再平均；
    - 测试集中没有 ≥4 分电影的用户无法计算 Recall（分母为 0），
      这些用户不进入均值，但单独统计数量。
    """
    per_user: list[dict] = []
    summary_rows: list[dict] = []

    for fold in folds:
        train_df, test_df = load_fold(data_dir, fold)
        model = CFModel(build_train_matrix(train_df, users, items))
        model.ensure_similarity(ALGO_USER)
        model.ensure_similarity(ALGO_ITEM)
        pop = popularity_scores(model)

        test_rel = (
            test_df[test_df["rating"] >= relevant_threshold]
            .groupby("user_id")["item_id"].apply(lambda s: set(int(x) for x in s))
            .to_dict()
        )
        test_users = {int(u) for u in test_df["user_id"].unique()}   # 该折测试集里出现过的用户
        id2idx = {int(m): i for i, m in enumerate(model.train.item_ids)}
        rec_items: dict[str, set[int]] = {ALGO_POPULAR: set(), ALGO_USER: set(), ALGO_ITEM: set()}
        n_no_test = 0        # 该折测试集里根本没有这个用户的评分（结构性缺席）
        n_no_positive = 0    # 有测试评分，但没有 ≥4 分的（无法定义 Recall 的分母）

        for u_idx in range(model.n_users):
            uid = int(model.train.user_ids[u_idx])
            seen = model.M[u_idx].astype(bool)
            if uid not in test_users:
                n_no_test += 1
                rel_ids = set()
            else:
                rel_ids = test_rel.get(uid, set())
            # 相关电影统一转到 item_idx 空间，并排除训练集已看过的（它们本来就不该被推荐）
            relevant = {id2idx[m] for m in rel_ids if m in id2idx and not seen[id2idx[m]]}
            if uid in test_users and not relevant:
                n_no_positive += 1

            recs = {
                ALGO_POPULAR: topk_from_scores(pop, seen, n),
                ALGO_USER: model.recommend(u_idx, ALGO_USER, K=K, n=n)["item_idx"].to_numpy(),
                ALGO_ITEM: model.recommend(u_idx, ALGO_ITEM, K=K, n=n)["item_idx"].to_numpy(),
            }
            for algo, rec in recs.items():
                rec_items[algo].update(int(x) for x in rec)
                if not relevant:
                    continue
                hits = len(set(int(x) for x in rec) & relevant)
                per_user.append({
                    "fold": fold, "algorithm": algo, "user_id": uid,
                    "n_relevant": len(relevant), "hits": hits,
                    "recall": round(hits / len(relevant), 4),
                    "precision": round(hits / n, 4),
                    "ndcg": round(ndcg_at_k(rec, relevant, n), 4),
                    "hit": int(hits > 0),
                })

        for algo in (ALGO_POPULAR, ALGO_USER, ALGO_ITEM):
            d = pd.DataFrame([r for r in per_user if r["fold"] == fold and r["algorithm"] == algo])
            summary_rows.append({
                "fold": fold, "algorithm": algo, "n_cutoff": n, "K": K,
                "users_evaluated": int(len(d)),
                "users_no_test": n_no_test,
                "users_no_positive": n_no_positive,
                "recall_mean": round(float(d["recall"].mean()), 4),
                "ndcg_mean": round(float(d["ndcg"].mean()), 4),
                "hitrate_mean": round(float(d["hit"].mean()), 4),
                "precision_mean": round(float(d["precision"].mean()), 4),
                "n_distinct_items": len(rec_items[algo]),
                "item_coverage": round(len(rec_items[algo]) / model.n_items, 4),
            })
        print(f"  fold {fold} 完成："
              + "，".join(f"{r['algorithm']} Hit@10={r['hitrate_mean']:.4f}"
                          for r in summary_rows if r["fold"] == fold))

    per_user_df = pd.DataFrame(per_user)
    fold_df = pd.DataFrame(summary_rows)
    summary = (
        fold_df.groupby("algorithm")
        .agg(
            folds=("fold", "count"),
            users_evaluated=("users_evaluated", "mean"),
            users_no_test=("users_no_test", "mean"),
            users_no_positive=("users_no_positive", "mean"),
            recall_mean=("recall_mean", "mean"), recall_std=("recall_mean", "std"),
            ndcg_mean=("ndcg_mean", "mean"), ndcg_std=("ndcg_mean", "std"),
            hitrate_mean=("hitrate_mean", "mean"), hitrate_std=("hitrate_mean", "std"),
            precision_mean=("precision_mean", "mean"),
            item_coverage_mean=("item_coverage", "mean"),
            item_coverage_std=("item_coverage", "std"),
            n_distinct_items_mean=("n_distinct_items", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return per_user_df, fold_df, summary


# --------------------------------------------------------------------- 作图
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c8d0da",
        "axes.labelcolor": "#33404f",
        "text.color": "#1e293b",
        "xtick.color": "#465264",
        "ytick.color": "#465264",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "font.size": 10,
    })
    return plt


PALETTE = {"baseline": "#94a3b8", "user": "#2563eb", "item": "#f59e0b"}
LABELS = {"baseline": "用户均值基准", "user": "User-CF", "item": "Item-CF"}


def fig_e1(summary: pd.DataFrame, out_dir: Path = FIG_DIR) -> list[Path]:
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    algos = ["baseline", "user", "item"]
    x = np.arange(len(algos))
    idx = summary.set_index("algorithm")
    ax = axes[0]
    vals = [idx.loc[a, "mae_mean"] for a in algos]
    errs = [idx.loc[a, "mae_std"] for a in algos]
    ax.bar(x, vals, yerr=errs, capsize=4, color=[PALETTE[a] for a in algos], width=0.58)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x, [LABELS[a] for a in algos])
    ax.set_title("MAE（越低越好）")
    ax.set_ylabel("MAE（纵轴截断）")
    ax.set_ylim(max(0, min(vals) - 0.06), max(vals) + max(errs) + 0.04)

    ax = axes[1]
    vals = [idx.loc[a, "rmse_mean"] for a in algos]
    errs = [idx.loc[a, "rmse_std"] for a in algos]
    ax.bar(x, vals, yerr=errs, capsize=4, color=[PALETTE[a] for a in algos], width=0.58)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.006, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x, [LABELS[a] for a in algos])
    ax.set_title("RMSE（越低越好）")
    ax.set_ylabel("RMSE（纵轴截断）")
    ax.set_ylim(min(vals) - 0.09, max(vals) + max(errs) + 0.05)
    fig.suptitle("E1 五折评测：误差均值 ± 标准差（5 个官方 fold）", fontsize=12)
    fig.tight_layout()
    p = out_dir / "e1_mae_rmse.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cf_algos = ["user", "item"]
    vals = [summary.set_index("algorithm").loc[a, "coverage_mean"] * 100 for a in cf_algos]
    ax.bar([0, 1], vals, color=[PALETTE[a] for a in cf_algos], width=0.5)
    for xi, v in zip([0, 1], vals):
        ax.text(xi, v + 1, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_xticks([0, 1], [LABELS[a] for a in cf_algos])
    ax.set_ylabel("测试评分中「用到 ≥1 个正相似邻居」的比例 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("E1 CF 有效邻居覆盖率（仅对 CF 定义）")
    ax.text(0.5, 4, "用户均值基准不需要邻居，不参与本指标", ha="center", fontsize=8.5, color="#64748b")
    fig.tight_layout()
    p = out_dir / "e1_coverage.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def fig_e2(df: pd.DataFrame, out_dir: Path = FIG_DIR) -> Path:
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for algo in ("user", "item"):
        d = df[df["algorithm"] == algo].sort_values("K")
        axes[0].plot(d["K"], d["rmse"], marker="o", color=PALETTE[algo], label=LABELS[algo])
        axes[1].plot(d["K"], d["coverage"] * 100, marker="o", color=PALETTE[algo], label=LABELS[algo])
    axes[0].set_title("E2 邻居数 K 与 RMSE")
    axes[0].set_xlabel("K（预测时参与的正相似邻居上限）")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)
    axes[1].set_title("E2 邻居数 K 与覆盖率")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("覆盖率 (%)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
    fig.suptitle("固定 u1.base / u1.test 的 K 敏感性实验", fontsize=12)
    fig.tight_layout()
    p = out_dir / "e2_k_sensitivity.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_e3(df: pd.DataFrame, out_dir: Path = FIG_DIR) -> Path:
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for algo in ("user", "item"):
        d = df[df["algorithm"] == algo].sort_values("target_ratio")
        xs = d["target_ratio"] * 100
        axes[0].plot(xs, d["rmse"], marker="o", color=PALETTE[algo], label=LABELS[algo])
        axes[1].plot(xs, d["coverage"] * 100, marker="o", color=PALETTE[algo], label=LABELS[algo])
    axes[0].set_title("E3 训练记录减少时的 RMSE")
    axes[0].set_xlabel("保留的训练记录比例 (%)")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("E3 训练记录减少时的覆盖率")
    axes[1].set_xlabel("保留的训练记录比例 (%)")
    axes[1].set_ylabel("覆盖率 (%)")
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)
    fig.suptitle("模拟实验：训练交互记录减少（u1，随机种子 42）", fontsize=12)
    fig.tight_layout()
    p = out_dir / "e3_sparsity.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


E4_LABELS = {"popular": "热门推荐", "user": "User-CF", "item": "Item-CF"}
E4_COLORS = {"popular": "#94a3b8", "user": "#2563eb", "item": "#f59e0b"}


def fig_e4(summary: pd.DataFrame, n: int = 10, out_dir: Path = FIG_DIR) -> Path:
    """Top-N 三项指标 + 推荐物品覆盖率，柱状图（误差棒为 5 折标准差）。"""
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = summary.set_index("algorithm")
    algos = ["popular", "user", "item"]
    metrics = [("recall_mean", "recall_std", f"Recall@{n}"),
               ("ndcg_mean", "ndcg_std", f"NDCG@{n}"),
               ("hitrate_mean", "hitrate_std", f"HitRate@{n}")]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    width = 0.24
    xs = np.arange(len(metrics))
    for j, algo in enumerate(algos):
        vals = [idx.loc[algo, m] for m, _, _ in metrics]
        errs = [0.0 if pd.isna(idx.loc[algo, e]) else idx.loc[algo, e] for _, e, _ in metrics]
        ax.bar(xs + (j - 1) * width, vals, width, yerr=errs, capsize=3,
               color=E4_COLORS[algo], label=E4_LABELS[algo])
        for xi, v, e in zip(xs + (j - 1) * width, vals, errs):
            ax.text(xi, v + e + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [t for _, _, t in metrics])
    ax.set_title(f"Top-{n} 推荐质量（候选 = 全部未评分电影，相关 = 测试评分 ≥4）")
    ax.set_ylabel("指标值（按用户平均）")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(idx[[m for m, _, _ in metrics]].to_numpy().max() * 1.35, 0.1))

    ax = axes[1]
    x2 = np.arange(len(algos))
    vals = [idx.loc[a, "item_coverage_mean"] * 100 for a in algos]
    errs = [idx.loc[a, "item_coverage_std"] * 100 for a in algos]
    ax.bar(x2, vals, 0.55, yerr=errs, capsize=4, color=[E4_COLORS[a] for a in algos])
    for xi, v in zip(x2, vals):
        ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x2, [E4_LABELS[a] for a in algos])
    ax.set_ylabel("推荐过的电影占目录比例 (%)")
    ax.set_title("推荐物品覆盖率：Top-10 触及多少部不同电影")
    ax.set_ylim(0, max(vals) * 1.25 + 2)
    fig.tight_layout()
    p = out_dir / "e4_topn.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- 扩展：Bias-MF
ALGO_MF = "mf"


def experiment_mf(
    data_dir: str | Path, users: np.ndarray, items: np.ndarray,
    folds: tuple[int, ...] = (1, 2, 3, 4, 5), k: int = 30, iterations: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """扩展实验：在同一批 fold、同一评分预测任务上评估 Bias-MF（ALS 求解）。"""
    from src.mf import BiasMF

    rows = []
    for fold in folds:
        train_df, test_df = load_fold(data_dir, fold)
        train = build_train_matrix(train_df, users, items)
        pairs = _index_pairs(test_df, users, items)
        actual = test_df["rating"].to_numpy(dtype=np.float64)
        mf = BiasMF(k=k, iterations=iterations).fit(train)
        preds = mf.predict_pairs(pairs)
        mae, rmse = pair_metrics(preds, actual)
        rows.append({
            "fold": fold, "algorithm": ALGO_MF, "k": k, "iterations": iterations,
            "mae": round(mae, 4), "rmse": round(rmse, 4),
            "train_seconds": round(mf.train_seconds, 2),
            "predict_seconds": 0.0, "n_pairs": int(len(pairs)),
        })
        print(f"  fold {fold} 完成：Bias-MF RMSE={rmse:.4f}（训练 {mf.train_seconds:.1f}s）")

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "algorithm": ALGO_MF, "folds": len(df), "k": k, "iterations": iterations,
        "mae_mean": round(df["mae"].mean(), 4), "mae_std": round(df["mae"].std(), 4),
        "rmse_mean": round(df["rmse"].mean(), 4), "rmse_std": round(df["rmse"].std(), 4),
        "train_seconds_mean": round(df["train_seconds"].mean(), 2),
    }])
    return df, summary
