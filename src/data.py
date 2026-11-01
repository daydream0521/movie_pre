"""MovieLens 100K 数据读取、官方五折切分与训练矩阵构建。

设计要点：
1. 训练矩阵 R 中 0 表示“未评分”，绝不表示 0 分；所有统计只对观测掩码 M 生效。
2. 用户/物品索引在整个数据集上一次性固定（943 x 1682），五个 fold 共享同一形状，
   这样相似度矩阵、电影元信息与索引可以稳定对应。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]
ITEM_COLS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "ml-100k"


def load_ratings(data_dir: str | Path | None = None) -> pd.DataFrame:
    """读取 u.data（制表符分隔的 10 万条评分）。"""
    path = Path(data_dir or default_data_dir()) / "u.data"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}。请先运行 `python scripts/download_data.py` 下载 MovieLens 100K。"
        )
    df = pd.read_csv(path, sep="\t", names=RATING_COLS, engine="python")
    for c in RATING_COLS:
        df[c] = df[c].astype("int64")
    return df


def load_movies(data_dir: str | Path | None = None) -> pd.DataFrame:
    """读取 u.item（latin-1 编码），解析片名与年份，附带 19 个类型标签。"""
    path = Path(data_dir or default_data_dir()) / "u.item"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先下载 MovieLens 100K。")
    df = pd.read_csv(path, sep="|", names=ITEM_COLS, encoding="latin-1", engine="python")
    df = df[ITEM_COLS].copy()  # 保持列序
    df["item_id"] = df["item_id"].astype("int64")

    # 片名末尾形如 "(1995)"，提取年份；个别条目缺失则置空
    year = df["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    df["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

    # 类型是 19 列 0/1 标志位，转成便于展示的列表
    genre_flags = df[GENRES].astype(int)
    df["genres"] = genre_flags.apply(
        lambda row: [g for g in GENRES if row[g] == 1 and g != "unknown"], axis=1
    )
    df["genres_text"] = df["genres"].apply(lambda gs: " / ".join(gs) if gs else "未标注")
    return df


def load_fold(data_dir: str | Path | None, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取官方第 fold 折（1..5）的 base/test。"""
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold 必须是 1..5")
    base = Path(data_dir or default_data_dir())
    train = pd.read_csv(base / f"u{fold}.base", sep="\t", names=RATING_COLS, engine="python")
    test = pd.read_csv(base / f"u{fold}.test", sep="\t", names=RATING_COLS, engine="python")
    for df in (train, test):
        for c in RATING_COLS:
            df[c] = df[c].astype("int64")
    return train, test


@dataclass
class TrainMatrix:
    """训练集评分矩阵（稀疏 + 稠密缓存）与相关统计量。"""

    R: sparse.csr_matrix          # 评分，0 表示缺失
    M: sparse.csr_matrix          # 观测掩码（0/1）
    user_ids: np.ndarray          # 行索引 -> user_id
    item_ids: np.ndarray          # 列索引 -> item_id
    global_mean: float

    @property
    def n_users(self) -> int:
        return self.R.shape[0]

    @property
    def n_items(self) -> int:
        return self.R.shape[1]

    @property
    def n_ratings(self) -> int:
        return int(self.M.sum())

    @property
    def density(self) -> float:
        return self.n_ratings / (self.n_users * self.n_items)

    @property
    def user_index(self) -> dict[int, int]:
        return {int(u): i for i, u in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[int, int]:
        return {int(m): i for i, m in enumerate(self.item_ids)}

    def user_means(self) -> np.ndarray:
        """用户均值：只对观测评分求平均；无评分用户回退全局均值。"""
        sums = np.asarray(self.R.sum(axis=1)).ravel()
        counts = np.asarray(self.M.sum(axis=1)).ravel()
        means = np.full(self.n_users, self.global_mean, dtype=np.float64)
        nz = counts > 0
        means[nz] = sums[nz] / counts[nz]
        return means

    def ratings_of(self, user_idx: int) -> pd.DataFrame:
        """某用户训练集中已评分的电影（item_idx, rating）。"""
        row = self.R.getrow(user_idx)
        return pd.DataFrame({"item_idx": row.indices, "rating": row.data})


def build_train_matrix(
    train: pd.DataFrame, all_users: np.ndarray, all_items: np.ndarray
) -> TrainMatrix:
    """用固定的用户/物品全集构建稀疏训练矩阵。"""
    uidx = {int(u): i for i, u in enumerate(all_users)}
    iidx = {int(m): i for i, m in enumerate(all_items)}
    rows = train["user_id"].map(uidx).to_numpy()
    cols = train["item_id"].map(iidx).to_numpy()
    vals = train["rating"].to_numpy(dtype=np.float32)
    R = sparse.csr_matrix((vals, (rows, cols)), shape=(len(all_users), len(all_items)))
    M = sparse.csr_matrix(
        (np.ones_like(vals, dtype=np.float32), (rows, cols)),
        shape=(len(all_users), len(all_items)),
    )
    return TrainMatrix(
        R=R, M=M, user_ids=np.asarray(all_users), item_ids=np.asarray(all_items),
        global_mean=float(train["rating"].mean()),
    )


def dataset_summary(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """数据统计，用于 README / 界面展示真实数据概况。"""
    counts = ratings.groupby("rating").size()
    return {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user_id"].nunique()),
        "n_items": int(ratings["item_id"].nunique()),
        "n_movies_meta": int(len(movies)),
        "density": float(len(ratings) / (ratings["user_id"].nunique() * movies["item_id"].nunique())),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_counts": {int(k): int(v) for k, v in counts.items()},
        "ratings_per_user_median": float(ratings.groupby("user_id").size().median()),
        "ratings_per_item_median": float(ratings.groupby("item_id").size().median()),
        "min_timestamp": int(ratings["timestamp"].min()),
        "max_timestamp": int(ratings["timestamp"].max()),
    }


def load_catalog(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """一次性加载评分、电影元信息，并返回全集的用户/物品 ID 数组。"""
    d = data_dir or default_data_dir()
    ratings = load_ratings(d)
    movies = load_movies(d)
    return ratings, movies, np.sort(ratings["user_id"].unique()), np.sort(ratings["item_id"].unique())
