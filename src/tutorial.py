"""协同过滤原理实验室：一张可手算的小评分矩阵与分步骤计算轨迹。

与真实实验的区别（刻意保留，并在界面写明）：
- 只有 5 用户 × 5 电影，为了让每一行算式都能被手算复核；
- 因此相似度不启用收缩、共同评分门槛降到 2（真实实验中是 3 + 收缩 n/(n+10)）。
除此之外，公式、中心化方式、Top-K 规则与 src/collaborative_filter.py 完全一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.collaborative_filter import ALGO_ITEM, ALGO_USER, CFModel
from src.data import TrainMatrix

TUTORIAL_USERS = ["Alice", "Bob", "Carol", "Dave", "Eve"]

TUTORIAL_MOVIES = [
    (1, "星际穿越", "Sci-Fi / Adventure"),
    (2, "盗梦空间", "Sci-Fi / Thriller"),
    (3, "爱在黎明破晓前", "Romance / Drama"),
    (4, "疾速追杀", "Action / Thriller"),
    (5, "小森林", "Drama"),
]

# 每行一个用户，None 表示“没看过 / 没评分”，绝不是 0 分
TUTORIAL_RATINGS = [
    [5, 4, 2, None, None],   # Alice：偏爱科幻，给文艺爱情片低分（目标：预测她对动作片的评分）
    [4, 4, 3, 5, None],      # Bob：口味与 Alice 接近，且看过疾速追杀
    [3, 2, 5, 2, 4],         # Carol：偏文艺，不喜欢动作片
    [2, 2, 5, 2, 5],         # Dave：重文艺、轻动作，与 Alice 口味相反
    [4, 4, 3, 4, 3],         # Eve：比较平均，略偏科幻
]

DEFAULT_TARGET_USER = 0   # Alice
DEFAULT_TARGET_ITEM = 3   # 疾速追杀（第 4 部）
DEFAULT_K = 3


def rating_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        TUTORIAL_RATINGS,
        index=TUTORIAL_USERS,
        columns=[f"T{tid} {title}" for tid, title, _ in TUTORIAL_MOVIES],
        dtype="float64",
    )
    return df


def movie_label(item_idx: int) -> str:
    tid, title, _ = TUTORIAL_MOVIES[item_idx]
    return f"T{tid}《{title}》"


def build_tutorial_model(min_common: int = 2, shrink: float = 0.0) -> tuple[CFModel, TrainMatrix]:
    """把小矩阵装进与真实实验相同的 TrainMatrix / CFModel，保证公式同源。"""
    rows, cols, vals = [], [], []
    for u, row in enumerate(TUTORIAL_RATINGS):
        for i, r in enumerate(row):
            if r is not None:
                rows.append(u)
                cols.append(i)
                vals.append(float(r))
    shape = (len(TUTORIAL_USERS), len(TUTORIAL_MOVIES))
    R = sparse.csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
    M = sparse.csr_matrix(
        (np.ones(len(vals), dtype=np.float32), (rows, cols)), shape=shape, dtype=np.float32
    )
    train = TrainMatrix(
        R=R, M=M,
        user_ids=np.arange(1, len(TUTORIAL_USERS) + 1),
        item_ids=np.array([m[0] for m in TUTORIAL_MOVIES]),
        global_mean=float(np.mean(vals)),
    )
    return CFModel(train, min_common=min_common, shrink=shrink), train


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def hand_calculation(algo: str, user_idx: int = DEFAULT_TARGET_USER,
                     item_idx: int = DEFAULT_TARGET_ITEM, K: int = DEFAULT_K) -> str:
    """生成可逐行复核的手算过程（数值全部现算，不硬编码）。"""
    model, _ = build_tutorial_model()
    ex = model.explain(user_idx, item_idx, algo, K=K)
    u_name = TUTORIAL_USERS[user_idx]
    m_name = movie_label(item_idx)

    if algo == ALGO_USER:
        lines = [f"目标：预测 **{u_name}** 对 {m_name} 的评分，K = {K}。", ""]
        lines.append(f"1) 用户均值：mean({u_name}) = {_fmt(ex.base)}")
        if ex.is_fallback:
            lines.append("2) 没有正相似邻居给过该片评分 → 回退到均值。")
            return "\n".join(lines)
        nb = ex.neighbors
        lines.append("2) 邻居（都对这部片评过分、且相似度为正）：")
        for _, r in nb.iterrows():
            name = TUTORIAL_USERS[int(r["neighbor_user_id"]) - 1]
            lines.append(
                f"   - {name}: sim = {_fmt(r['similarity'])}, 给该片 {r['rating']:.0f} 分, "
                f"mean = {_fmt(r['neighbor_mean'])}"
            )
        lines.append(f"3) 分母 Σ|sim| = {_fmt(ex.sim_sum)}")
        terms = " + ".join(
            f"{_fmt(r['similarity'])}×({r['rating']:.0f}-{_fmt(r['neighbor_mean'])})"
            for _, r in nb.iterrows()
        )
        num = sum(r["similarity"] * (r["rating"] - r["neighbor_mean"]) for _, r in nb.iterrows())
        lines.append(f"4) 分子 = {terms} = {_fmt(num)}")
        lines.append(
            f"5) 预测 = {_fmt(ex.base)} + {_fmt(num)} / {_fmt(ex.sim_sum)} = **{_fmt(ex.pred)}**"
        )
    else:
        lines = [f"目标：预测 **{u_name}** 对 {m_name} 的评分，K = {K}。", ""]
        lines.append(f"1) 用户均值：mean({u_name}) = {_fmt(ex.base)}")
        if ex.is_fallback:
            lines.append("2) 该用户评过的电影与目标电影都不正相关 → 回退到均值。")
            return "\n".join(lines)
        nb = ex.neighbors
        lines.append(f"2) {u_name} 评过、且与{m_name}相似的电影（相似度为正）：")
        for _, r in nb.iterrows():
            j = int(np.where([m[0] for m in TUTORIAL_MOVIES] == r["neighbor_item_id"])[0][0])
            lines.append(
                f"   - {movie_label(j)}: sim = {_fmt(r['similarity'])}, "
                f"{u_name} 打了 {r['user_rating']:.0f} 分"
            )
        lines.append(f"3) 分母 Σ|sim| = {_fmt(ex.sim_sum)}")
        terms = " + ".join(
            f"{_fmt(r['similarity'])}×({r['user_rating']:.0f}-{_fmt(ex.base)})" for _, r in nb.iterrows()
        )
        num = sum(r["similarity"] * (r["user_rating"] - ex.base) for _, r in nb.iterrows())
        lines.append(f"4) 分子 = {terms} = {_fmt(num)}")
        lines.append(
            f"5) 预测 = {_fmt(ex.base)} + {_fmt(num)} / {_fmt(ex.sim_sum)} = **{_fmt(ex.pred)}**"
        )
    return "\n".join(lines)


def steps(algo: str, user_idx: int = DEFAULT_TARGET_USER,
          item_idx: int = DEFAULT_TARGET_ITEM, K: int = DEFAULT_K) -> list[dict]:
    """分步骤的讲解轨迹：每一步给出表格数据与一句结论。"""
    model, train = build_tutorial_model()
    u_name = TUTORIAL_USERS[user_idx]
    m_name = movie_label(item_idx)
    ex = model.explain(user_idx, item_idx, algo, K=K)
    mat = rating_frame()

    steps_list: list[dict] = [{
        "title": "第 1 步：评分矩阵长什么样",
        "body": (
            f"行是用户、列是电影，空白代表**没有评分**（不是 0 分）。"
            f"现在我们想预测 {u_name} 对 {m_name} 的评分——格子里是空的。"
        ),
        "visual": "matrix",
        "data": {"matrix": mat, "user": user_idx, "item": item_idx},
    }]

    if algo == ALGO_USER:
        rated = [i for i, r in enumerate(TUTORIAL_RATINGS[user_idx]) if r is not None]
        sim = model.ensure_similarity(ALGO_USER)
        candidates = [v for v in range(len(TUTORIAL_USERS)) if v != user_idx and TUTORIAL_RATINGS[v][item_idx] is not None]
        rows = []
        for v in range(len(TUTORIAL_USERS)):
            if v == user_idx:
                continue
            common = [i for i in range(len(TUTORIAL_MOVIES))
                      if TUTORIAL_RATINGS[user_idx][i] is not None and TUTORIAL_RATINGS[v][i] is not None]
            rows.append({
                "用户": TUTORIAL_USERS[v],
                "共同评分电影": "、".join(f"T{i+1}" for i in common) if common else "无",
                "共同评分数": len(common),
                "相似度 sim": round(float(sim[user_idx, v]), 3),
                "是否给目标片评过分": "是" if TUTORIAL_RATINGS[v][item_idx] is not None else "否",
            })
        sim_df = pd.DataFrame(rows).sort_values("相似度 sim", ascending=False)

        steps_list.append({
            "title": "第 2 步：谁和我口味相似？",
            "body": (
                f"先把每个人的评分减去他自己的均值（消除“有人总打高分”的偏差），再算余弦相似度。"
                f"只有**给目标电影评过分**的人，才能提供参考分数——{u_name} 自己评过的电影是 "
                + "、".join(f"T{i+1}" for i in rated) + "。"
            ),
            "visual": "similarity_table",
            "data": {"table": sim_df, "algo": algo},
        })

        pos = sim_df[sim_df["相似度 sim"] > 0]
        topk = pos.head(K)
        steps_list.append({
            "title": f"第 3 步：取 Top-{K} 个正相似邻居",
            "body": (
                f"负相似度不能用来外推（口味相反的人打 2 分，不代表你也会打 2 分），所以只用正值。"
                f"K = {K} 表示**最多**取 {K} 人，这里实际取到 {len(topk)} 人："
                + ("、".join(topk["用户"].tolist()) if len(topk) else "无") + "。"
            ),
            "visual": "topk",
            "data": {"table": topk, "K": K, "available": len(pos), "algo": algo},
        })

        if ex.is_fallback:
            steps_list.append({
                "title": "第 4 步：没有可用邻居 → 均值回退",
                "body": f"没有任何正相似邻居给 {m_name} 打过分，预测值直接取 {u_name} 的均值 {_fmt(ex.base)}。",
                "visual": "result", "data": {"explanation": ex},
            })
        else:
            nb = ex.neighbors.assign(
                用户=lambda d: [TUTORIAL_USERS[int(x) - 1] for x in d["neighbor_user_id"]]
            )[["用户", "similarity", "neighbor_mean", "rating", "contribution"]]
            nb.columns = ["用户", "相似度 sim", "TA 的均值", "TA 给该片的评分", "对预测的贡献"]
            steps_list.append({
                "title": "第 4 步：加权求和得到预测",
                "body": (
                    f"每个人给出的“偏差”是 (评分 − TA 的均值)，按相似度加权平均，再加回 {u_name} 自己的均值。"
                    f"分母是 Σ|sim| = {_fmt(ex.sim_sum)}，它保证权重归一。"
                ),
                "visual": "weighted",
                "data": {"table": nb, "explanation": ex},
            })
    else:
        rated = [i for i, r in enumerate(TUTORIAL_RATINGS[user_idx]) if r is not None]
        sim = model.ensure_similarity(ALGO_ITEM)
        rows = []
        for j in rated:
            rows.append({
                "候选电影": movie_label(j),
                f"{u_name} 的评分": TUTORIAL_RATINGS[user_idx][j],
                "共同评分用户数": int((model.M[:, j] & model.M[:, item_idx]).sum()),
                "与目标片相似度": round(float(sim[item_idx, j]), 3),
            })
        sim_df = pd.DataFrame(rows).sort_values("与目标片相似度", ascending=False)

        steps_list.append({
            "title": "第 2 步：我看过的片里，哪部和它像？",
            "body": (
                f"这里换一个方向：不再找相似的人，而是找**相似的电影**。"
                f"候选只能是 {u_name} 已经看过的电影（"
                + "、".join(f"T{i+1}" for i in rated) + "），因为它们才能提供“其实我喜欢什么”的证据。"
            ),
            "visual": "similarity_table",
            "data": {"table": sim_df, "algo": algo},
        })

        topk = sim_df[sim_df["与目标片相似度"] > 0].head(K)
        steps_list.append({
            "title": f"第 3 步：取 Top-{K} 部相似电影",
            "body": (
                f"“相似”指**被同一批用户以相似方式评价**（调整余弦：先减去每个用户的评分均值），"
                f"不是类型标签相同。实际取到 {len(topk)} 部。"
            ),
            "visual": "topk",
            "data": {"table": topk, "K": K, "available": int((sim_df["与目标片相似度"] > 0).sum()), "algo": algo},
        })

        if ex.is_fallback:
            steps_list.append({
                "title": "第 4 步：没有可用相似电影 → 均值回退",
                "body": f"预测值退化为 {u_name} 的均值 {_fmt(ex.base)}。",
                "visual": "result", "data": {"explanation": ex},
            })
        else:
            name_by_id = {m[0]: m for m in TUTORIAL_MOVIES}
            nb = ex.neighbors.copy()
            nb["电影"] = [movie_label(list(name_by_id).index(int(x))) for x in nb["neighbor_item_id"]]
            nb = nb[["电影", "similarity", "user_rating", "centered_rating", "contribution"]]
            nb.columns = ["相似电影", "相似度 sim", "该用户评分", "评分−均值", "对预测的贡献"]
            steps_list.append({
                "title": "第 4 步：按相似度加权得到预测",
                "body": (
                    f"每部相似电影告诉我们“用户对它的偏好偏离均值多少”，乘以相似度求和，"
                    f"再除以 Σ|sim| = {_fmt(ex.sim_sum)}，最后加回用户均值。"
                ),
                "visual": "weighted",
                "data": {"table": nb, "explanation": ex},
            })

    steps_list.append({
        "title": "第 5 步：结论与容易被追问的点",
        "body": (
            f"预测评分 **{_fmt(ex.pred)}**。"
            "注意三件事：① 空白从不参与计算，不是 0 分；② 相似度只来自“行为共现”，"
            "两部类型完全不同的片子也可能高度相似；③ 若没有可用邻居，界面会标注“均值回退”，"
            "那不是协同过滤的结果。"
        ),
        "visual": "result", "data": {"explanation": ex},
    })
    return steps_list
