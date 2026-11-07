"""E5：NCF 风格显式评分预测的五折实验。

运行（必须用装好 PyTorch 的环境）：
    D:/BaiduNetdiskDownload/envs/deeplearning/python.exe scripts/run_e5.py

产物：
    results/e5_fold_results.csv    每折指标、最佳轮次、训练耗时
    results/e5_curves.csv          每折每轮的训练/验证 RMSE
    results/e5_summary.csv         五折均值 ± 标准差
    results/e5_comparison.csv      与 User-CF / Item-CF / Bias-MF 的统一对照
    results/figures/e5_learning_curves.png
    results/figures/e5_compare.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_catalog, load_fold  # noqa: E402
from src.evaluation import (  # noqa: E402
    FIG_DIR, LABELS, PALETTE, RESULTS_DIR, pair_metrics, setup_matplotlib,
)
from src.ncf import TrainConfig, predict, train_fold  # noqa: E402

NCF_COLOR = "#a855f7"
NCF_LABEL = "NCF 风格 MLP"


def _index_map(ids: np.ndarray) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(ids)}


def run(folds: tuple[int, ...], cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, users, items = load_catalog()
    umap, imap = _index_map(users), _index_map(items)

    fold_rows: list[dict] = []
    curve_rows: list[dict] = []

    for fold in folds:
        t0 = time.perf_counter()
        train_df, test_df = load_fold(ROOT / "data" / "ml-100k", fold)
        # np.array 复制一份，避免 pandas 返回只读视图触发 torch 警告
        u_tr = np.array(train_df["user_id"].map(umap), dtype=np.int64)
        i_tr = np.array(train_df["item_id"].map(imap), dtype=np.int64)
        r_tr = np.array(train_df["rating"], dtype=np.float32)

        res = train_fold(u_tr, i_tr, r_tr, len(users), len(items), cfg, seed_offset=fold)

        u_te = np.array(test_df["user_id"].map(umap), dtype=np.int64)
        i_te = np.array(test_df["item_id"].map(imap), dtype=np.int64)
        preds = predict(res["model"], u_te, i_te)
        mae, rmse = pair_metrics(preds, test_df["rating"].to_numpy(dtype=np.float64))

        fold_rows.append({
            "fold": fold,
            "algorithm": "ncf",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "best_epoch": res["best_epoch"],
            "epochs_run": res["epochs_run"],
            "best_val_rmse": round(res["best_val_rmse"], 4),
            "train_seconds": round(res["train_seconds"], 1),
            "n_train": res["n_train"],
            "n_val": res["n_val"],
            "device": res["device"],
        })
        for h in res["history"]:
            curve_rows.append({"fold": fold, **h})
        print(f"  fold {fold} 完成：MAE={mae:.4f} RMSE={rmse:.4f} "
              f"best_epoch={res['best_epoch']}/{res['epochs_run']} "
              f"耗时 {res['train_seconds']:.1f}s（{time.perf_counter() - t0:.1f}s 含读盘）")

    return pd.DataFrame(fold_rows), pd.DataFrame(curve_rows)


def summarize(fold_df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    return pd.DataFrame([{
        "algorithm": "ncf",
        "folds": len(fold_df),
        "mae_mean": round(fold_df["mae"].mean(), 4),
        "mae_std": round(fold_df["mae"].std(), 4),
        "rmse_mean": round(fold_df["rmse"].mean(), 4),
        "rmse_std": round(fold_df["rmse"].std(), 4),
        "train_seconds_mean": round(fold_df["train_seconds"].mean(), 1),
        "best_epoch_mean": round(fold_df["best_epoch"].mean(), 1),
        "emb_dim": cfg.emb_dim,
        "hidden": "x".join(str(h) for h in cfg.hidden),
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "device": fold_df["device"].iloc[0],
    }])


def compare(ncf_summary: pd.DataFrame) -> pd.DataFrame:
    """把 E1 的基准/CF、Bias-MF 与 E5 拼成一张对照表（只读已有结果，不重算）。"""
    rows = []
    e1_path, mf_path = RESULTS_DIR / "e1_summary.csv", RESULTS_DIR / "mf_summary.csv"
    if e1_path.exists():
        e1 = pd.read_csv(e1_path, encoding="utf-8-sig").set_index("algorithm")
        rows += [
            {"model": "用户均值基准", "mae": e1.loc["baseline", "mae_mean"], "mae_std": e1.loc["baseline", "mae_std"],
             "rmse": e1.loc["baseline", "rmse_mean"], "rmse_std": e1.loc["baseline", "rmse_std"],
             "train_seconds": 0.0, "note": "不使用邻居，无训练"},
            {"model": "User-CF", "mae": e1.loc["user", "mae_mean"], "mae_std": e1.loc["user", "mae_std"],
             "rmse": e1.loc["user", "rmse_mean"], "rmse_std": e1.loc["user", "rmse_std"],
             "train_seconds": e1.loc["user", "build_seconds_mean"], "note": "K=30，建相似度即可预测"},
            {"model": "Item-CF", "mae": e1.loc["item", "mae_mean"], "mae_std": e1.loc["item", "mae_std"],
             "rmse": e1.loc["item", "rmse_mean"], "rmse_std": e1.loc["item", "rmse_std"],
             "train_seconds": e1.loc["item", "build_seconds_mean"], "note": "K=30，建相似度即可预测"},
        ]
    if mf_path.exists():
        mf = pd.read_csv(mf_path, encoding="utf-8-sig").iloc[0]
        rows.append({"model": "Bias-MF", "mae": mf["mae_mean"], "mae_std": mf["mae_std"],
                     "rmse": mf["rmse_mean"], "rmse_std": mf["rmse_std"],
                     "train_seconds": mf["train_seconds_mean"], "note": "ALS，k=30，15 轮"})
    n = ncf_summary.iloc[0]
    rows.append({"model": NCF_LABEL, "mae": n["mae_mean"], "mae_std": n["mae_std"],
                 "rmse": n["rmse_mean"], "rmse_std": n["rmse_std"],
                 "train_seconds": n["train_seconds_mean"],
                 "note": f"emb{n['emb_dim']}+MLP{n['hidden']}，EarlyStopping，{n['device']}"})
    return pd.DataFrame(rows)


def fig_curves(curves: pd.DataFrame, out_dir: Path = FIG_DIR, best_epoch: float | None = None) -> Path:
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    # 第 1 轮预测接近 0、RMSE 在 2.0 以上，会把纵轴拉爆；从第 2 轮起画，才能看清收敛与过拟合
    d = curves[curves["epoch"] >= 2]
    fig, ax = plt.subplots(figsize=(7.8, 4.3))

    for _, g in d.groupby("fold"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g["train_rmse"], color="#94a3b8", lw=0.9, alpha=0.45)
        ax.plot(g["epoch"], g["val_rmse"], color=NCF_COLOR, lw=0.9, alpha=0.35)

    agg = d.groupby("epoch").agg(
        tr=("train_rmse", "mean"),
        va=("val_rmse", "mean"), va_lo=("val_rmse", "min"), va_hi=("val_rmse", "max"),
    ).reset_index()
    ax.plot(agg["epoch"], agg["tr"], color="#475569", lw=2.4, label="训练集 RMSE（折均值）")
    ax.plot(agg["epoch"], agg["va"], color=NCF_COLOR, lw=2.4, label="验证集 RMSE（折均值）")
    ax.fill_between(agg["epoch"], agg["va_lo"], agg["va_hi"], color=NCF_COLOR, alpha=0.13, lw=0)
    lo = min(agg["tr"].min(), agg["va"].min())
    hi = max(agg["tr"].max(), agg["va"].max())

    if best_epoch:
        ax.axvline(best_epoch, color="#94a3b8", ls=":", lw=1.2)
        ax.annotate(f"验证损失最低点：平均第 {best_epoch:.1f} 轮", xy=(best_epoch, lo),
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5, color="#475569")

    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE（1–5 星）")
    ax.set_title("E5 NCF 风格 MLP：训练/验证损失（细线为各折，从第 2 轮起）")
    ax.set_ylim(lo - 0.008, hi + 0.015)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    p = out_dir / "e5_learning_curves.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_compare(cmp: pd.DataFrame, out_dir: Path = FIG_DIR) -> Path:
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [PALETTE["baseline"], PALETTE["user"], PALETTE["item"], "#a855f7", "#7c3aed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax, (col, err, title) in zip(
        axes, [("rmse", "rmse_std", "RMSE（越低越好）"), ("mae", "mae_std", "MAE（越低越好）")]
    ):
        vals = cmp[col].to_numpy()
        errs = cmp[err].to_numpy()
        xs = np.arange(len(cmp))
        ax.bar(xs, vals, yerr=errs, capsize=4, color=colors[:len(cmp)], width=0.6)
        for x, v, e in zip(xs, vals, errs):
            ax.text(x, v + e + 0.01, f"{v:.4f}", ha="center", fontsize=8.5)
        ax.set_xticks(xs, cmp["model"], rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(min(vals) - 0.12, max(vals + errs) + 0.06)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.suptitle("E5 统一对照：显式评分预测的五折均值 ± 标准差", fontsize=12)
    fig.tight_layout()
    p = out_dir / "e5_compare.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(max_epochs=args.epochs, patience=args.patience,
                      device=args.device, seed=args.seed)
    t0 = time.perf_counter()
    print(f"== E5：NCF 风格显式评分预测（{len(args.folds)} 折）==")
    fold_df, curves = run(tuple(args.folds), cfg)
    summary = summarize(fold_df, cfg)
    cmp = compare(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(RESULTS_DIR / "e5_fold_results.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(RESULTS_DIR / "e5_curves.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / "e5_summary.csv", index=False, encoding="utf-8-sig")
    cmp.to_csv(RESULTS_DIR / "e5_comparison.csv", index=False, encoding="utf-8-sig")

    fig_curves(curves, best_epoch=float(summary.iloc[0]['best_epoch_mean']))
    fig_compare(cmp)

    print(f"E5 用时 {time.perf_counter() - t0:.1f}s")
    print(summary.to_string(index=False))
    print()
    print(cmp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
