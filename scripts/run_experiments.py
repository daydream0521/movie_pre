"""E1 / E2 / E3 批量实验入口：跑完把真实结果写进 results/。

用法：
    python scripts/run_experiments.py            # 跑全部 E1/E2/E3
    python scripts/run_experiments.py --only e2  # 只跑 E2
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_catalog  # noqa: E402
from src.evaluation import (  # noqa: E402
    FIG_DIR, RESULTS_DIR, experiment_e1, experiment_e2, experiment_e3,
    fig_e1, fig_e2, fig_e3,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["e1", "e2", "e3", "all"], default="all")
    ap.add_argument("--K", type=int, default=30, help="E1/E3 使用的邻居数")
    ap.add_argument("--fold", type=int, default=1, help="E2/E3 固定使用的 fold")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _, _, users, items = load_catalog()

    t_all = time.perf_counter()
    meta = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": platform.processor() or "unknown",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "K_default": args.K,
        "fold_for_e2_e3": args.fold,
    }
    pd.Series(meta).to_csv(RESULTS_DIR / "run_meta.csv", header=["value"])

    if args.only in ("all", "e1"):
        print("== E1 五折算法比较 ==")
        t = time.perf_counter()
        df, summary = experiment_e1(ROOT / "data" / "ml-100k", users, items, K=args.K)
        df.to_csv(RESULTS_DIR / "e1_fold_results.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(RESULTS_DIR / "e1_summary.csv", index=False, encoding="utf-8-sig")
        fig_e1(summary)
        print(f"\nE1 用时 {time.perf_counter()-t:.1f}s")
        print(summary.to_string(index=False))

    if args.only in ("all", "e2"):
        print("\n== E2 邻居数 K 的影响（固定 u1）==")
        t = time.perf_counter()
        df2 = experiment_e2(ROOT / "data" / "ml-100k", users, items, fold=args.fold)
        df2.to_csv(RESULTS_DIR / "e2_k_sensitivity.csv", index=False, encoding="utf-8-sig")
        fig_e2(df2)
        print(f"E2 用时 {time.perf_counter()-t:.1f}s")
        print(df2.to_string(index=False))

    if args.only in ("all", "e3"):
        print("\n== E3 训练记录减少的模拟实验（固定 u1）==")
        t = time.perf_counter()
        df3 = experiment_e3(ROOT / "data" / "ml-100k", users, items, K=args.K, fold=args.fold)
        df3.to_csv(RESULTS_DIR / "e3_sparsity.csv", index=False, encoding="utf-8-sig")
        fig_e3(df3)
        print(f"E3 用时 {time.perf_counter()-t:.1f}s")
        print(df3.to_string(index=False))

    print(f"\n全部完成，总用时 {time.perf_counter()-t_all:.1f}s；结果目录：{RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
