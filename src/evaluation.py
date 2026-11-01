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
            "coverage": round(res.coverage, 4),
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
    vals = [summary.set_index("algorithm").loc[a, "coverage_mean"] * 100 for a in algos]
    ax.bar(x, vals, color=[PALETTE[a] for a in algos], width=0.58)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, [LABELS[a] for a in algos])
    ax.set_ylabel("测试评分覆盖率 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("E1 预测覆盖率：至少有一个有效相似邻居的测试评分占比")
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
