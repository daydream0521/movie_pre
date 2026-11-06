"""MovieLens Explorer — 推荐系统课堂演示应用。

运行：streamlit run app.py
所有推荐/指标均由真实 MovieLens 100K 训练数据现算，无预置结果。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.collaborative_filter import ALGO_ITEM, ALGO_USER, CFModel
from src.data import (
    GENRES, build_train_matrix, dataset_summary, load_catalog, load_fold,
)
from src.demo import render_demo
from src.tutorial import (
    DEFAULT_K, DEFAULT_TARGET_ITEM, DEFAULT_TARGET_USER, TUTORIAL_MOVIES, TUTORIAL_USERS,
    hand_calculation, movie_label, rating_frame, steps as tutorial_steps,
    target_is_rated, true_rating,
)
from src.ui import (
    ACCENT, ACCENT_2, GOOD, WARN, hero, inject_css, kpi_row, matrix_table, movie_card,
    note, section, step_progress,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ml-100k"
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="MovieLens Explorer · 协同过滤",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()


# --------------------------------------------------------------------- 缓存
@st.cache_data(show_spinner=False)
def get_catalog():
    ratings, movies, users, items = load_catalog(DATA_DIR)
    summary = dataset_summary(ratings, movies)
    return ratings, movies, users, items, summary


@st.cache_resource(show_spinner="正在用 u1.base 构建相似度矩阵…")
def get_fold_model(fold: int = 1):
    _, _, users, items, _ = get_catalog()
    train_df, test_df = load_fold(DATA_DIR, fold)
    mat = build_train_matrix(train_df, users, items)
    model = CFModel(mat)
    model.ensure_similarity(ALGO_USER)
    model.ensure_similarity(ALGO_ITEM)
    return model


@st.cache_data(show_spinner=False)
def get_movie_meta() -> dict:
    _, movies, _, _, _ = get_catalog()
    return {
        int(r["item_id"]): {
            "title": r["clean_title"], "year": r["year"],
            "genres": r["genres_text"], "genres_list": r["genres"],
        }
        for _, r in movies.iterrows()
    }


@st.cache_data(show_spinner=False)
def read_results(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


def _numeric_format(df: pd.DataFrame) -> dict:
    """评分列显示整数，其余浮点列保留 3 位小数。"""
    fmt = {}
    for c in df.columns:
        if df[c].dtype.kind == "f":
            fmt[c] = "{:.0f}" if ("评分" in str(c) and "均值" not in str(c) and "贡献" not in str(c)) else "{:.3f}"
    return fmt


def dark(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b8c6ce", family="Segoe UI, Microsoft YaHei", size=12),
        margin=dict(l=8, r=8, t=36, b=8),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
    )
    if fig.layout.title.text:
        fig.update_layout(title=dict(font=dict(size=13, color="#edf1f2")))
    else:
        fig.update_layout(margin=dict(l=8, r=8, t=14, b=8))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# --------------------------------------------------------------------- 页面：概览
def page_overview():
    ratings, movies, users, items, summary = get_catalog()
    hero(
        "推荐系统如何猜你喜欢",
        "电影评分里的推荐原理",
        "用 10 万条真实电影评分，演示“推荐系统如何猜你喜欢”。"
        "所有相似度、预测分和推荐列表都由当前训练数据现算，界面里能看到每一步的加权来源。",
    )
    kpi_row([
        ("评分数", f"{summary['n_ratings']:,}", "MovieLens 100K，1–5 分整数"),
        ("用户", f"{summary['n_users']:,}", "每人中位 65 条评分"),
        ("电影", f"{summary['n_items']:,}", "19 个类型标签"),
        ("矩阵密度", f"{summary['density']*100:.2f}%", "其余 93.7% 是空白，不是 0 分"),
    ])

    c1, c2 = st.columns([1.15, 1])
    with c1:
        section("评分分布", "全部 10 万条评分")
        dist = pd.Series(summary["rating_counts"]).sort_index()
        fig = go.Figure(go.Bar(
            x=[f"{k} 分" for k in dist.index], y=dist.values,
            marker_color=[ACCENT if k < 4 else GOOD for k in dist.index],
            text=dist.values, textposition="outside",
        ))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, float(dist.max()) * 1.16])
        st.plotly_chart(dark(fig, 300), use_container_width=True)
        note("评分整体偏高（均值 3.53），所以“减去每个用户自己的均值”是必要的——否则爱打高分的用户会影响相似度。")

    with c2:
        section("两种算法在回答什么问题", "")
        st.markdown(
            f"""
<div class="kpi" style="margin-bottom:10px">
  <div class="label">USER-CF</div>
  <div class="value" style="font-size:1rem">“和我口味相似的人，怎么评价这部电影？”</div>
  <div class="hint">相似用户 → 他们对目标电影的评分</div>
</div>
<div class="kpi">
  <div class="label">ITEM-CF</div>
  <div class="value" style="font-size:1rem">“我看过的电影里，哪部被大家以相似方式评价？”</div>
  <div class="hint">相似电影 → 我对那些电影的评分</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("两者都只使用用户–物品的**交互行为**，不是电影类型关键词匹配。")

    section("数据集概况", "官方 u.data / u.item")
    rows = [
        ("评分条数", f"{summary['n_ratings']:,}"),
        ("用户数 / 电影数", f"{summary['n_users']} / {summary['n_items']}"),
        ("评分均值", f"{summary['rating_mean']:.4f}"),
        ("每用户评分中位数", f"{int(summary['ratings_per_user_median'])}"),
        ("每电影评分中位数", f"{int(summary['ratings_per_item_median'])}"),
        ("密度", f"{summary['density']*100:.2f}%"),
        ("官方五折", "u1–u5，每折约 80% 训练 / 20% 测试"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["项目", "数值"]), hide_index=True, use_container_width=True)

    section("怎么用这个应用", "")
    st.markdown(
        "- **原理实验室**：5×5 小矩阵，一步一步看相似度、Top-K 邻居和加权求和，数值可手算复核。\n"
        "- **电影推荐**：选一个真实用户，切换 User-CF / Item-CF，看 Top-10 以及每一条推荐的依据。\n"
        "- **实验结果**：E1 五折对比、E2 邻居数 K、E3 训练记录减少，全部来自 `results/*.csv` 的实际运行。\n"
        "- **从 CF 到深度学习**：相似度 → 潜在向量 → Embedding+MLP 的概念过渡。"
    )


# --------------------------------------------------------------------- 页面：原理实验室
def page_lab():
    hero(
        "协同过滤原理实验室",
        "step by step · 5 × 5 可手算",
        "一张 5 名用户 × 5 部电影的小矩阵，空白代表没有评分。"
        "按“下一步”逐步推进，每一步都给出真实算出来的数字。",
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.1])
    algo_label = c1.radio("算法", ["User-CF", "Item-CF"], horizontal=True, key="lab_algo")
    algo = ALGO_USER if algo_label == "User-CF" else ALGO_ITEM
    user_idx = c2.selectbox("目标用户", range(len(TUTORIAL_USERS)),
                            format_func=lambda i: TUTORIAL_USERS[i],
                            index=DEFAULT_TARGET_USER, key="lab_user")
    item_idx = c3.selectbox("目标电影", range(len(TUTORIAL_MOVIES)),
                            format_func=movie_label, index=DEFAULT_TARGET_ITEM, key="lab_item")
    K = c4.slider("邻居数 K", 1, 4, DEFAULT_K, key="lab_k")

    # 目标用户已经给目标电影打过分时，把这条评分临时隐藏，否则「已知答案还问预测」没有意义
    hide = (user_idx, item_idx) if target_is_rated(user_idx, item_idx) else None
    if hide is not None:
        note(
            f"<b>{TUTORIAL_USERS[user_idx]}</b> 在原始矩阵中已经给 {movie_label(item_idx)} 打过分"
            f"（{true_rating(user_idx, item_idx):.0f} 分）。为了演示“预测”，这一格已临时从训练矩阵中移除："
            "用户均值和相似度都会按隐藏后的矩阵重算，最后可以和真实评分对比。",
            warn=True,
        )

    steps = tutorial_steps(algo, user_idx, item_idx, K, hide=hide)
    state_key = f"lab_step_{algo}_{user_idx}_{item_idx}_{K}"
    if state_key not in st.session_state:
        for k in [k for k in st.session_state if k.startswith("lab_step_")]:
            del st.session_state[k]
        st.session_state[state_key] = 0
    step = st.session_state[state_key]

    st.markdown(step_progress(step, len(steps)), unsafe_allow_html=True)

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("← 上一步", disabled=step == 0, use_container_width=True):
        st.session_state[state_key] = max(0, step - 1)
        st.rerun()
    if b2.button("下一步 →", disabled=step >= len(steps) - 1, use_container_width=True):
        st.session_state[state_key] = min(len(steps) - 1, step + 1)
        st.rerun()
    if b3.button("重置", use_container_width=True):
        st.session_state[state_key] = 0
        st.rerun()

    s = steps[step]
    section(f"{s['title']}", f"第 {step + 1} / {len(steps)} 步")
    st.markdown(s["body"])

    visual = s["visual"]
    data = s["data"]

    if visual == "matrix":
        st.markdown(matrix_table(data["matrix"], data["user"], data["item"], data.get("hidden", False)),
                    unsafe_allow_html=True)
        st.caption(
            "行 = 用户，列 = 电影；“—” 表示矩阵里没有这条评分（缺失，不是 0 分）；"
            "蓝色行是目标用户，紫色列是目标电影。"
            + ("问号格是真实存在、但被临时隐藏的评分。" if data.get("hidden") else "问号格就是要预测的位置。")
        )

    elif visual in ("similarity_table", "topk"):
        df = data["table"]
        st.dataframe(df.style.format(_numeric_format(df)), hide_index=True, use_container_width=True)
        if visual == "topk":
            st.caption(f"K = {data['K']}，当前有 {data['available']} 个正相似候选，实际使用 {len(df)} 个。")
        else:
            st.caption("相似度 = 中心化向量之间的余弦；负值意味着口味相反，预测时不使用。")

    elif visual == "weighted":
        ex = data["explanation"]
        df = data["table"]
        st.dataframe(df.style.format(_numeric_format(df)), hide_index=True, use_container_width=True)
        contrib = df.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=contrib[df.columns[-1]], y=contrib[df.columns[0]], orientation="h",
            marker_color=[ACCENT if v >= 0 else "#ef4444" for v in contrib[df.columns[-1]]],
        ))
        fig.update_layout(title="每个邻居对预测的贡献（分）", showlegend=False)
        st.plotly_chart(dark(fig, 260), use_container_width=True)

    elif visual == "result":
        ex = data["explanation"]
        badge = "均值回退" if ex.is_fallback else "协同过滤预测"
        color = WARN if ex.is_fallback else GOOD
        st.markdown(
            f'<div class="kpi" style="max-width:460px">'
            f'<div class="label">{badge}</div>'
            f'<div class="value" style="font-size:2.1rem;color:{color}">{ex.pred:.3f}</div>'
            f'<div class="hint">mean(u) = {ex.base:.3f}　|　Σ|sim| = {ex.sim_sum:.3f}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("手算复核（4–5 行看完这个预测）", expanded=(step == len(steps) - 1)):
        st.markdown(hand_calculation(algo, user_idx, item_idx, K, hide=hide))

    with st.expander("老师可能会问", expanded=False):
        st.markdown(
            """
**1. 空白格子能当成 0 分吗？**
不能。空白只说明「这条评分没有被记录」，它既可能是没看过，也可能是看过但没评、甚至是不想评。当成 0 分等于替用户断言「极度讨厌」，用户均值会被拉低，相似度也会失真。所以计算只统计双方都有评分的位置。

**2. 为什么要减去用户均值（中心化）？**
有人习惯全打 4–5 分，有人只打 2–3 分。中心化后比较的是“相对于他自己，这部片算好还是差”，相似度才反映口味而不是打分习惯。

**3. 相似度为什么会出现负数？**
中心化向量的夹角超过 90°，说明这两人口味相反。**本实现选择不使用负相似邻居**：在 `Σ|sim|` 归一化下，负权重会让“邻居低于他自己均值”变成“预测高于均值”，容易把结果推到区间外。这是设计选择而不是数学必需——若保留负相似，需要重新讨论归一化方式并接受更多越界预测。

**4. 共同评分为 1 部电影时，相似度能信吗？**
不能。一部片的巧合不足以说明口味，真实实验中要求共同评分 ≥ 3，并用 n/(n+10) 收缩相似度（本页小矩阵为便于手算关闭了这两项）。

**5. User-CF 和 Item-CF 的差别到底在哪？**
User-CF 找“人”（谁给我参考分），Item-CF 找“物”（我喜欢的片还像哪部）。前者依赖用户数量，后者依赖物品数量；用户口味漂移时 Item-CF 通常更稳。
            """
        )


def _rank_of(rec: pd.DataFrame | None, item_idx: int) -> int | None:
    """某部电影在推荐列表中的名次（1 起），不在列表里返回 None。"""
    if rec is None or rec.empty:
        return None
    hit = np.flatnonzero(rec["item_idx"].to_numpy() == item_idx)
    return int(hit[0]) + 1 if hit.size else None


def _mean_gap(rec_u: pd.DataFrame, rec_i: pd.DataFrame, common: set[int]) -> float:
    """两种算法在共同推荐条目上的平均预测分歧。"""
    if not common:
        return float("nan")
    a = rec_u[rec_u["item_idx"].isin(common)].set_index("item_idx")["pred"]
    b = rec_i[rec_i["item_idx"].isin(common)].set_index("item_idx")["pred"]
    return float((a - b).abs().mean())


def _pick_random_user(all_users: list[int], key: str = "rec_user_select") -> None:
    """button 的 on_click 回调：在脚本执行前改 selectbox 的 session_state。"""
    current = st.session_state.get(key)
    pool = [u for u in all_users if u != current] or all_users
    st.session_state[key] = random.choice(pool)


# --------------------------------------------------------------------- 页面：电影推荐
def _rec_cards(df: pd.DataFrame, model, meta: dict, empty_hint: str = ""):
    """Top-N 电影卡片：徽标是推荐分（排序依据），小字给出预测评分与邻居数。"""
    if df.empty:
        if empty_hint:
            st.caption(empty_hint)
        return
    cols = st.columns(5)
    for i, (_, r) in enumerate(df.iterrows()):
        item_id = int(model.train.item_ids[int(r["item_idx"])])
        m = meta[item_id]
        nbr_txt = f"{int(r['n_neighbors'])} 个邻居" if not r["is_fallback"] else "无邻居"
        cols[i % 5].markdown(
            movie_card(
                m["title"], m["year"], m["genres"], float(r["pred"]),
                meta=f"预测 {r['pred']:.2f} · {nbr_txt}",
                rank=i + 1, fallback=bool(r["is_fallback"]), item_id=item_id,
                score_text="回退" if r["is_fallback"] else f"{r['rank_score']:.2f}",
            ),
            unsafe_allow_html=True,
        )


def _explain_block(model, meta: dict, u_idx: int, item_idx: int, algo: str, K: int,
                   heading: str | None = None, show_chart: bool = True) -> None:
    """渲染单条推荐的加权依据：来源表 + 贡献条形图。"""
    ex = model.explain(u_idx, item_idx, algo, K=K)
    item_id = int(model.train.item_ids[item_idx])
    m = meta[item_id]
    if heading:
        st.markdown(f"**{heading}**")

    st.markdown(
        f'<div class="kpi" style="margin-bottom:8px">'
        f'<div class="label">{m["genres"]}</div>'
        f'<div class="value" style="font-size:1.05rem">{m["title"]}'
        f'{" (" + str(int(m["year"])) + ")" if pd.notna(m["year"]) else ""}</div>'
        f'<div class="hint">推荐分 {ex.rank_score:.3f}　|　预测 {ex.pred:.3f}　|　'
        f'mean(u) {ex.base:.3f}　|　Σ|sim| {ex.sim_sum:.3f}'
        f'{"　|　均值回退" if ex.is_fallback else ""}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(ex.formula)

    if ex.neighbors.empty:
        note("这个目标没有可用的正相似邻居，预测退化为该用户的训练均值，不能当作协同过滤的结果。", warn=True)
        return

    nb = ex.neighbors.copy()
    if algo == ALGO_USER:
        nb["来源"] = ["用户 " + str(int(x)) for x in nb["neighbor_user_id"]]
        nb = nb[["来源", "similarity", "rating", "contribution"]]
        nb.columns = ["来源（相似用户）", "相似度", "TA 给这部片的评分", "对预测的贡献"]
        contrib_col, label_col = "对预测的贡献", "来源（相似用户）"
    else:
        nb["来源"] = [meta[int(x)]["title"] for x in nb["neighbor_item_id"]]   # neighbor_item_id 已是电影 id
        nb = nb[["来源", "similarity", "user_rating", "contribution"]]
        nb.columns = ["来源（相似电影）", "相似度", "你给它的评分", "对预测的贡献"]
        contrib_col, label_col = "对预测的贡献", "来源（相似电影）"

    st.dataframe(nb.style.format({
        "相似度": "{:.4f}", "对预测的贡献": "{:+.4f}",
        "TA 给这部片的评分": "{:.0f}", "你给它的评分": "{:.0f}",
    }), hide_index=True, use_container_width=True)

    if show_chart:
        cnt = nb.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=cnt[contrib_col], y=cnt[label_col], orientation="h",
            marker_color=[GOOD if v >= 0 else "#ef4444" for v in cnt[contrib_col]],
        ))
        fig.update_layout(showlegend=False, title="各邻居对预测分的贡献（已除以 Σ|sim|）")
        st.plotly_chart(dark(fig, max(200, 24 * len(cnt))), use_container_width=True)


def page_recommend():
    ratings, movies, users, items, summary = get_catalog()
    meta = get_movie_meta()
    model = get_fold_model(1)
    train_counts = np.asarray(model.M.sum(axis=1)).ravel()

    hero(
        "电影推荐体验",
        "real data · u1.base 训练",
        "选一位真实用户，用训练集里他的历史评分生成 Top-10，并展开任意一条的加权依据。"
        "切到“双算法对比”，可以直接看到 User-CF 与 Item-CF 给出的不同结果。",
    )

    all_users = [int(u) for u in users]
    c1, c2, c3, c4, c5 = st.columns([1.6, 0.55, 1.0, 0.85, 0.85])
    uid = c1.selectbox(
        "选择一个用户", all_users, index=0,
        format_func=lambda u: f"用户 {u} · u1.base 中 {int(train_counts[int(np.where(model.train.user_ids == u)[0][0])])} 条评分",
        key="rec_user_select",
    )
    # 随机切换必须改 selectbox 自己的 key，并在回调里赋值（脚本运行后再改会抛异常）
    c2.button(
        "🎲 随机", on_click=_pick_random_user, args=(all_users,), use_container_width=True,
        help="随机换一个用户，方便现场对比不同人的推荐结果",
    )
    K = c4.select_slider("邻居数 K", options=[5, 10, 20, 30, 50], value=30, key="rec_k")
    topn = c5.selectbox("推荐数量", [5, 10, 20], index=1, key="rec_n")

    view = st.radio("视图", ["单算法", "双算法对比", "同一部电影，两种解释"],
                    horizontal=True, key="rec_view",
                    help="对比视图会同时计算两种算法，适合回答“它们到底差在哪”")
    compare = view != "单算法"
    need_both = view in ("双算法对比", "同一部电影，两种解释")

    algo_label = st.radio(
        "算法", ["User-CF", "Item-CF"], horizontal=True, key="rec_algo", disabled=compare,
        help=None if not compare else "对比视图会同时跑两种算法，此处不生效",
    )
    algo = ALGO_USER if algo_label == "User-CF" else ALGO_ITEM

    u_idx = int(np.where(model.train.user_ids == uid)[0][0])
    n_train_ratings = int(model.M[u_idx].sum())
    rec_u = model.recommend(u_idx, ALGO_USER, K=K, n=topn) if (need_both or algo == ALGO_USER) else None
    rec_i = model.recommend(u_idx, ALGO_ITEM, K=K, n=topn) if (need_both or algo == ALGO_ITEM) else None
    rec_main = rec_u if algo == ALGO_USER else rec_i
    stats = model.neighbor_stats(u_idx, algo)

    kpi_row([
        ("该用户训练集评分", f"{n_train_ratings}", "u1.base 中出现过的记录"),
        ("全库矩阵密度", f"{summary['density']*100:.2f}%", f"{model.n_users}×{model.n_items} 个格子"),
        ("可用正相似" + ("用户" if algo == ALGO_USER else "电影"), f"{stats['positive_neighbors']}", "相似度 > 0 的候选"),
        ("候选电影", f"{model.n_items - n_train_ratings}", "已排除训练集看过的片"),
        ("回退条数", f"{int(rec_main['is_fallback'].sum())}/{len(rec_main)}", "无邻居时取用户均值"),
    ])

    left, right = st.columns([1.05, 1])
    with left:
        section("他/她给过的高分电影", "训练集中评分 ≥ 4.5 的代表作")
        seen = model.train.ratings_of(u_idx)
        seen = seen[seen["rating"] >= 4.5]
        top_seen = seen.head(8)
        cols = st.columns(4)
        for i, (_, r) in enumerate(top_seen.iterrows()):
            mid = int(model.train.item_ids[int(r["item_idx"])])
            m = meta[mid]
            cols[i % 4].markdown(
                movie_card(m["title"], m["year"], m["genres"], float(r["rating"]),
                           meta=f"真实评分 {r['rating']:.0f} 分", item_id=mid,
                           score_text=f"{r['rating']:.0f} 分"),
                unsafe_allow_html=True,
            )
        if top_seen.empty:
            st.caption("该用户在训练集中没有 4.5 分以上的记录。")

        section("评分分布", f"用户 {uid} 在训练集中的全部评分")
        hist = np.bincount(model.X[u_idx][model.M[u_idx]].astype(int), minlength=6)[1:6]
        fig = go.Figure(go.Bar(x=[f"{i} 分" for i in range(1, 6)], y=hist,
                               marker_color=[ACCENT, ACCENT, "#60a5fa", GOOD, GOOD]))
        fig.update_layout(showlegend=False, title=f"均值 {model.mu[u_idx]:.2f}，共 {n_train_ratings} 条")
        st.plotly_chart(dark(fig, 240), use_container_width=True)

    with right:
        section("类型偏好", "按该用户给过 ≥4 分的电影统计")
        rated_idx = np.flatnonzero(model.M[u_idx])
        liked = [i for i in rated_idx if model.X[u_idx, i] >= 4]
        counter: dict[str, int] = {}
        for i in liked:
            for g in meta[int(model.train.item_ids[i])]["genres_list"]:
                counter[g] = counter.get(g, 0) + 1
        gser = pd.Series(counter).sort_values(ascending=False).head(8)
        if len(gser):
            fig = go.Figure(go.Bar(x=gser.values, y=gser.index, orientation="h", marker_color=ACCENT_2))
            fig.update_layout(showlegend=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(dark(fig, 240), use_container_width=True)
        else:
            st.caption("该用户没有 4 分以上的评分。")

        section("排序分与预测分的区别", "")
        note(
            "列表按 <b>推荐分</b>（相似度加权和 Σ sim·偏差）排序；卡片小字里的 <b>预测</b> "
            "是同一组邻居的加权平均（+ 用户均值）。"
            "用加权平均排序会让“只有 1 个弱邻居、恰好打 5 分”的冷门片被推到榜首，"
            "所以 Top-N 用加权和、评分预测用加权平均。"
        )

    if view == "单算法":
        section(f"Top-{topn} 推荐", f"{algo_label} · K = {K} · 已排除训练集评过的 {n_train_ratings} 部")
        _rec_cards(rec_main, model, meta)
        st.caption(
            "徽标是**推荐分**（相似度加权和，用来排序，不是 1–5 的评分）；"
            "卡片小字里的“预测”才是同一组邻居的加权平均分。"
        )
        capped = int((rec_main["pred"] >= 4.995).sum())
        if capped:
            st.caption(
                f"其中 {capped} 条的预测评分触顶 5.00（邻居都给了 5 分，加权后超过上限被裁剪）；"
                "排序不受裁剪影响，仍按推荐分。"
            )

        section("查看单条推荐的依据", "选一部，看加权过程")
        options = [
            f"#{i+1}  {meta[int(model.train.item_ids[int(r['item_idx'])])]['title']}"
            f"（推荐分 {r['rank_score']:.2f}）"
            for i, (_, r) in enumerate(rec_main.iterrows())
        ]
        pick = st.selectbox("推荐条目", options, index=0, key="rec_pick")
        item_idx = int(rec_main.iloc[options.index(pick)]["item_idx"])
        _explain_block(model, meta, u_idx, item_idx, algo, K)

        with st.expander("为什么同一条推荐在不同 K 下会变化？", expanded=False):
            st.markdown(
                "K 限制的是**预测时实际参与的正相似邻居数量**。K 小 → 只用最像的几个，方差大但更贴合；"
                "K 大 → 平均更稳，但会把“没那么像”的人也算进来，推荐分随之变化。"
                "卡片上的“N 个邻居”就是这个条目实际用到的邻居数。"
            )

    elif view == "双算法对比":
        set_u = set(rec_u["item_idx"].tolist())
        set_i = set(rec_i["item_idx"].tolist())
        common = set_u & set_i
        union = sorted(set_u | set_i,
                       key=lambda x: min(_rank_of(rec_u, x) or 999, _rank_of(rec_i, x) or 999))
        jac = len(common) / max(1, len(set_u | set_i))
        kpi_row([
            ("共同推荐", f"{len(common)}", f"两个列表重合 {len(common)}/{topn}，Jaccard {jac:.2f}"),
            ("只有 User-CF 推", f"{len(set_u - set_i)}", "按“相似的人”选出来的"),
            ("只有 Item-CF 推", f"{len(set_i - set_u)}", "按“相似的片”选出来的"),
            ("预测分歧（共同项）", f"{_mean_gap(rec_u, rec_i, common):.2f}", "同一部片两种算法的预测分差"),
        ])

        section("Top 推荐并排对比", f"K = {K} · 同一候选集（已排除训练集评过的 {n_train_ratings} 部）")
        cu, ci = st.columns(2)
        with cu:
            st.markdown(f"**User-CF** · 找相似的人")
            _rec_cards(rec_u.head(5), model, meta)
        with ci:
            st.markdown(f"**Item-CF** · 找相似的片")
            _rec_cards(rec_i.head(5), model, meta)
        st.caption(
            "两边各展示前 5 名。徽标是推荐分（排序依据），小字里的“预测”是评分预测；"
            "同一部片在两个列表里的名次往往不同——这正是“找相似的人”和“找相似的片”的差别。"
        )

        section("逐条差异", "同一部电影在两个列表里的位置")
        dict_u = {int(r["item_idx"]): r for _, r in rec_u.iterrows()}
        dict_i = {int(r["item_idx"]): r for _, r in rec_i.iterrows()}
        rows = []
        for x in union:
            mid = int(model.train.item_ids[x])
            ru_, ri_ = _rank_of(rec_u, x), _rank_of(rec_i, x)
            rows.append({
                "电影": meta[mid]["title"],
                "User-CF 排名": "—" if ru_ is None else str(ru_),
                "Item-CF 排名": "—" if ri_ is None else str(ri_),
                "User-CF 推荐分": f"{dict_u[x]['rank_score']:.3f}" if x in dict_u else "—",
                "Item-CF 推荐分": f"{dict_i[x]['rank_score']:.3f}" if x in dict_i else "—",
                "User-CF 预测": f"{dict_u[x]['pred']:.2f}" if x in dict_u else "—",
                "Item-CF 预测": f"{dict_i[x]['pred']:.2f}" if x in dict_i else "—",
            })
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)
        st.caption(
            "“—”表示这部片没有进入该算法的 Top-N。排名差得越多，说明两种“相似”的定义在这部片上分歧越大。"
        )

        section("同一部电影，两种解释", "两边都用各自的邻居拆解")
        opt = [
            f"{meta[int(model.train.item_ids[int(x)])]['title']}"
            f"（User-CF #{_rank_of(rec_u, x) or '—'} / Item-CF #{_rank_of(rec_i, x) or '—'}）"
            for x in union
        ]
        pick = st.selectbox("选择一部电影", opt, index=0, key="rec_cmp_pick")
        item_idx = int(union[opt.index(pick)])
        e1, e2 = st.columns(2)
        with e1:
            _explain_block(model, meta, u_idx, item_idx, ALGO_USER, K,
                           heading="User-CF：哪些相似用户影响了它", show_chart=False)
        with e2:
            _explain_block(model, meta, u_idx, item_idx, ALGO_ITEM, K,
                           heading="Item-CF：哪些相似电影影响了它", show_chart=False)

        with st.expander("为什么两个列表差异这么大？", expanded=False):
            st.markdown(
                "- User-CF 只参考**和你相似的人**给这部片的分数，所以结果集中在“同类人群都爱看”的电影；\n"
                "- Item-CF 只参考**你看过的电影**与目标的相似度，所以结果更贴近你已有的观看轨迹；\n"
                "- 两者都不使用类型标签，重合度高低本身就是一个可以讨论的观察点："
                "重合越多说明“人群共识”和“个人轨迹”越一致。"
            )

    else:  # 同一部电影，两种解释
        section("挑一部你还没看过的电影", "两种算法分别解释为什么推荐它")
        cand = np.flatnonzero(~model.M[u_idx])
        cand_items = [int(model.train.item_ids[i]) for i in cand]
        default_id = int(model.train.item_ids[int(rec_main.iloc[0]["item_idx"])])
        mid = st.selectbox(
            "电影", cand_items, index=cand_items.index(default_id),
            format_func=lambda x: meta[x]["title"],
            key="rec_same_movie",
        )
        item_idx = int(np.where(model.train.item_ids == mid)[0][0])
        ru = _rank_of(rec_u, item_idx)
        ri = _rank_of(rec_i, item_idx)
        st.caption(f"在 User-CF 的 Top-{topn} 里排 #{ru or '—'}，在 Item-CF 的 Top-{topn} 里排 #{ri or '—'}。")
        e1, e2 = st.columns(2)
        with e1:
            _explain_block(model, meta, u_idx, item_idx, ALGO_USER, K,
                           heading="User-CF：找相似的人", show_chart=True)
        with e2:
            _explain_block(model, meta, u_idx, item_idx, ALGO_ITEM, K,
                           heading="Item-CF：找相似的片", show_chart=True)

# --------------------------------------------------------------------- 页面：实验结果
def page_results():
    hero(
        "实验结果",
        "e1 · e2 · e3 · e4",
        "全部指标由 scripts/run_experiments.py 在官方五折上真实跑出，页面只读取 results/*.csv。"
        "图表不写死结论，只标注怎么读。",
    )
    e1 = read_results("e1_fold_results.csv")
    s1 = read_results("e1_summary.csv")
    e2 = read_results("e2_k_sensitivity.csv")
    e3 = read_results("e3_sparsity.csv")
    e4 = read_results("e4_summary.csv")
    e4_fold = read_results("e4_fold_results.csv")

    if e1 is None:
        note("还没有实验输出。请在项目根目录运行：<code>python scripts/run_experiments.py</code>", warn=True)
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["E1 五折算法比较", "E2 邻居数 K", "E3 训练记录减少", "E4 Top-10 推荐质量"]
    )

    with tab1:
        idx = s1.set_index("algorithm")
        order = ["baseline", "user", "item"]
        names = {"baseline": "用户均值基准", "user": "User-CF", "item": "Item-CF"}
        kpi_row([
            (names[a], f"{idx.loc[a, 'rmse_mean']:.4f}",
             f"RMSE ± {idx.loc[a, 'rmse_std']:.4f}　MAE {idx.loc[a, 'mae_mean']:.4f}")
            for a in order
        ])

        c1, c2 = st.columns([1.25, 1])
        with c1:
            fig = go.Figure()
            for a, color in zip(order, ["#94a3b8", ACCENT, "#f59e0b"]):
                fig.add_trace(go.Bar(
                    name=names[a], x=["MAE", "RMSE"],
                    y=[idx.loc[a, "mae_mean"], idx.loc[a, "rmse_mean"]],
                    marker_color=color,
                    error_y=dict(type="data", array=[idx.loc[a, "mae_std"], idx.loc[a, "rmse_std"]], visible=True),
                ))
            fig.update_layout(barmode="group", title="五折平均误差 ± 标准差（越低越好）")
            st.plotly_chart(dark(fig, 320), use_container_width=True)
        with c2:
            fig = go.Figure()
            cf_order = ["user", "item"]
            for a, color in zip(cf_order, [ACCENT, "#f59e0b"]):
                fig.add_trace(go.Bar(name=names[a], x=[names[a]], y=[idx.loc[a, "coverage_mean"] * 100],
                                     marker_color=color, text=[f"{idx.loc[a, 'coverage_mean']*100:.2f}%"],
                                     textposition="outside"))
            fig.update_layout(showlegend=False,
                              title="CF 有效邻居覆盖率（仅对 CF 定义）",
                              yaxis_title="%", yaxis_range=[0, 108])
            st.plotly_chart(dark(fig, 320), use_container_width=True)
        st.caption(
            "覆盖率 = 测试评分中「至少用到一个正相似邻居完成预测」的比例（= 1 − 回退率），只对 CF 有意义；"
            "用户均值基准不需要邻居，因此不参与这项比较，也不应把它的 100% 拿来当上限。"
        )

        section("逐折明细", "5 个官方 fold")
        show = e1[["fold", "algorithm", "mae", "rmse", "coverage", "fallback_rate", "mean_neighbors",
                   "build_seconds", "predict_seconds"]].copy()
        show["algorithm"] = show["algorithm"].map(names)
        show["coverage"] = show["coverage"].fillna(-1).map(
            lambda v: "不适用" if v == -1 else f"{v*100:.2f}%"
        )
        show["fallback_rate"] = (show["fallback_rate"] * 100).map(lambda v: f"{v:.2f}%")
        st.dataframe(show, hide_index=True, use_container_width=True)

        # 依据实测数据生成的结论（不是写死的说法）
        item_wins = int((e1[e1.algorithm == "item"].set_index("fold")["rmse"]
                         < e1[e1.algorithm == "user"].set_index("fold")["rmse"]).sum())
        better = "User-CF" if idx.loc["user", "rmse_mean"] < idx.loc["item", "rmse_mean"] else "Item-CF"
        section("怎么读这张表", "")
        st.markdown(
            f"- 5 个 fold 的 RMSE 标准差约 {idx.loc['user','rmse_std']:.3f}–{idx.loc['item','rmse_std']:.3f}，"
            f"折间波动大于两种算法之间的差距，所以结论要看**均值 + 标准差**，不要只看某一折。\n"
            f"- 平均 RMSE 更低的是 **{better}**（{idx.loc['user' if better=='User-CF' else 'item','rmse_mean']:.4f} vs "
            f"{idx.loc['item' if better=='User-CF' else 'user','rmse_mean']:.4f}）；"
            f"Item-CF 在 {item_wins}/5 折上优于 User-CF。\n"
            f"- 两种 CF 都明显优于用户均值基准（RMSE {idx.loc['baseline','rmse_mean']:.4f}），"
            f"说明相似度确实带来了额外信息。\n"
            f"- 覆盖率：User-CF {idx.loc['user','coverage_mean']*100:.2f}% / "
            f"Item-CF {idx.loc['item','coverage_mean']*100:.2f}%，"
            f"说明约 {idx.loc['item','fallback_rate_mean']*100:.2f}% 的 Item-CF 预测没有邻居、退回了均值。\n"
            f"- 构建相似度只需 {idx.loc['user','build_seconds_mean']:.3f}s（用户）/ "
            f"{idx.loc['item','build_seconds_mean']:.3f}s（物品），2 万条测试评分的批量预测约 "
            f"{idx.loc['user','predict_seconds_mean']:.2f}s —— 这是纯 CPU 的量级，不含 GPU。"
        )

    with tab2:
        if e2 is None:
            note("E2 结果缺失，请运行 <code>python scripts/run_experiments.py --only e2</code>", warn=True)
        else:
            c1, c2 = st.columns(2)
            fig = go.Figure()
            for a, color in zip(["user", "item"], [ACCENT, "#f59e0b"]):
                d = e2[e2.algorithm == a].sort_values("K")
                fig.add_trace(go.Scatter(x=d["K"], y=d["rmse"], mode="lines+markers",
                                         name=names[a], line=dict(color=color, width=2.5),
                                         marker=dict(size=8)))
            fig.update_layout(title="K 与 RMSE", xaxis_title="K", yaxis_title="RMSE")
            c1.plotly_chart(dark(fig, 330), use_container_width=True)

            fig = go.Figure()
            for a, color in zip(["user", "item"], [ACCENT, "#f59e0b"]):
                d = e2[e2.algorithm == a].sort_values("K")
                fig.add_trace(go.Scatter(x=d["K"], y=d["coverage"] * 100, mode="lines+markers",
                                         name=names[a], line=dict(color=color, width=2.5),
                                         marker=dict(size=8)))
            fig.update_layout(title="K 与覆盖率", xaxis_title="K", yaxis_title="%")
            c2.plotly_chart(dark(fig, 330), use_container_width=True)

            best = e2.loc[e2.groupby("algorithm")["rmse"].idxmin()]
            txt = "；".join(f"{names[r['algorithm']]} 在 K={int(r['K'])} 时 RMSE 最低（{r['rmse']:.4f}）"
                           for _, r in best.iterrows())
            du = e2[e2.algorithm == "user"].sort_values("K")
            di = e2[e2.algorithm == "item"].sort_values("K")
            cov_span = max(du["coverage"].max() - du["coverage"].min(),
                           di["coverage"].max() - di["coverage"].min())
            tail = "；".join(
                f"{names[a]} 在 K={int(d['K'].iloc[-1])} 时 RMSE 比最低点高 {d['rmse'].iloc[-1] - d['rmse'].min():+.4f}"
                for a, d in (("user", du), ("item", di))
            )
            st.markdown(f"- {txt}。\n")
            st.markdown(
                "- 曲线是**实测**的：K 从最小值增加到中等时 RMSE 下降较快，说明邻居太少会让少数极端邻居主导预测；"
                f"再继续增大时变化很小（{tail}），并入较弱邻居的代价被大量平均抵消。\n"
            )
            if cov_span < 0.005:
                st.markdown(
                    f"- 覆盖率几乎不随 K 变化（波动 {cov_span*100:.2f} 个百分点）："
                    "回退只发生在“一个提供证据的正相似邻居都没有”时，K 再大也补不出来。"
                    "这一点与“K 越大覆盖率越高”的直觉不同。"
                )
            else:
                st.markdown(
                    f"- 覆盖率随 K 上升（波动 {cov_span*100:.2f} 个百分点）：K 越大越容易凑够邻居。"
                )
            st.dataframe(e2, hide_index=True, use_container_width=True)

    with tab3:
        if e3 is None:
            note("E3 结果缺失，请运行 <code>python scripts/run_experiments.py --only e3</code>", warn=True)
        else:
            c1, c2 = st.columns(2)
            for col, metric, title, scale in [(c1, "rmse", "保留比例与 RMSE", 1), (c2, "coverage", "保留比例与覆盖率", 100)]:
                fig = go.Figure()
                for a, color in zip(["user", "item"], [ACCENT, "#f59e0b"]):
                    d = e3[e3.algorithm == a].sort_values("target_ratio")
                    fig.add_trace(go.Scatter(x=d["target_ratio"] * 100, y=d[metric] * scale,
                                             mode="lines+markers", name=names[a],
                                             line=dict(color=color, width=2.5), marker=dict(size=8)))
                fig.update_layout(title=title, xaxis_title="保留的训练记录比例 (%)",
                                  yaxis_title="RMSE" if metric == "rmse" else "%")
                col.plotly_chart(dark(fig, 330), use_container_width=True)

            base = e3[(e3.target_ratio == 1.0)].set_index("algorithm")["rmse"]
            last = e3[(e3.target_ratio == e3.target_ratio.min())].set_index("algorithm")["rmse"]
            st.markdown(
                f"- 训练记录保留到 {e3.target_ratio.min()*100:.0f}% 时，User-CF RMSE 由 "
                f"{base['user']:.4f} 变为 {last['user']:.4f}，Item-CF 由 {base['item']:.4f} 变为 {last['item']:.4f}。\n"
                "- 这是**训练交互记录减少的模拟实验**，不是严格的冷启动实验：用户集合没有变化，"
                "只是每人留下的评分更少。\n"
                "- 测试集固定为 u1.test，过程中没有从测试集获取任何训练信息。"
            )
            st.dataframe(e3, hide_index=True, use_container_width=True)

    with tab4:
        if e4 is None:
            note("E4 结果缺失，请运行 <code>python scripts/run_experiments.py --only e4</code>", warn=True)
        else:
            idx4 = e4.set_index("algorithm")
            names4 = {"popular": "热门推荐", "user": "User-CF", "item": "Item-CF"}
            colors4 = {"popular": "#94a3b8", "user": ACCENT, "item": "#f59e0b"}
            order4 = ["popular", "user", "item"]
            kpi_row([
                (names4[a], f"Recall@10 {idx4.loc[a, 'recall_mean']:.4f}",
                 f"NDCG {idx4.loc[a, 'ndcg_mean']:.4f}　Hit {idx4.loc[a, 'hitrate_mean']:.4f}")
                for a in order4
            ])

            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig = go.Figure()
                for m, label in [("recall_mean", "Recall@10"), ("ndcg_mean", "NDCG@10"),
                                 ("hitrate_mean", "HitRate@10")]:
                    fig.add_trace(go.Bar(
                        name=label, x=[names4[a] for a in order4],
                        y=[idx4.loc[a, m] for a in order4],
                        marker_color=[colors4[a] for a in order4], showlegend=(m == "recall_mean"),
                    ))
                fig.update_layout(barmode="group", title="Top-10 推荐质量（按用户平均，5 折均值）")
                st.plotly_chart(dark(fig, 330), use_container_width=True)
            with c2:
                fig = go.Figure(go.Bar(
                    x=[names4[a] for a in order4],
                    y=[idx4.loc[a, "item_coverage_mean"] * 100 for a in order4],
                    marker_color=[colors4[a] for a in order4],
                    text=[f"{idx4.loc[a, 'item_coverage_mean']*100:.1f}%" for a in order4],
                    textposition="outside",
                ))
                fig.update_layout(showlegend=False, title="推荐物品覆盖率（触及目录的比例）",
                                  yaxis_title="%", yaxis_range=[0, max(
                                      idx4["item_coverage_mean"].max() * 100 * 1.3, 5)])
                st.plotly_chart(dark(fig, 330), use_container_width=True)

            best_recall = idx4["recall_mean"].idxmax()
            best_ndcg = idx4["ndcg_mean"].idxmax()
            st.markdown(
                f"- 相关定义：测试集中评分 **≥4** 的电影；候选集：**全部未在训练集出现的电影**（三个模型完全一致）；"
                f"截断 N=10。\n"
                f"- Recall@10 最高的是 **{names4[best_recall]}**（{idx4.loc[best_recall, 'recall_mean']:.4f}），"
                f"NDCG@10 最高的是 **{names4[best_ndcg]}**（{idx4.loc[best_ndcg, 'ndcg_mean']:.4f}）。"
                f"数值本身很小是正常的：每个用户的测试正例只有十几部片，10 个推荐位能覆盖的比例天然有限。\n"
                f"- 热门推荐只推了 **{idx4.loc['popular', 'n_distinct_items_mean']:.0f}** 部不同的电影"
                f"（覆盖 {idx4.loc['popular', 'item_coverage_mean']*100:.1f}% 目录），"
                f"Item-CF 推了 **{idx4.loc['item', 'n_distinct_items_mean']:.0f}** 部"
                f"（{idx4.loc['item', 'item_coverage_mean']*100:.1f}%）：命中率接近，但热门推荐的多样性几乎为零。\n"
                f"- 每折有约 {float(e4['users_no_test'].mean()):.0f} 名用户在该折测试集里没有任何评分，"
                f"另有 {float(e4['users_no_positive'].mean()):.0f} 名用户有测试评分但没有 ≥4 分的电影——"
                f"后者无法计算 Recall（分母为 0），两者都不进入均值。"
            )

            section("逐折明细", "5 个官方 fold")
            if e4_fold is not None:
                show4 = e4_fold[["fold", "algorithm", "users_evaluated", "users_no_test",
                                 "users_no_positive", "recall_mean", "ndcg_mean", "hitrate_mean",
                                 "item_coverage"]].copy()
                show4["algorithm"] = show4["algorithm"].map(names4)
                st.dataframe(show4, hide_index=True, use_container_width=True)

            note(
                "<b>这个指标不能直接当“推荐好不好”看</b>：只把测试集里评过 ≥4 分的电影当作相关，"
                "没出现过的电影既不知道是否喜欢、也不能算负反馈；而且测试评分只覆盖每折的一部分用户。"
                "它衡量的是“能不能在 10 个位置里捞到已知的好片”，不是平台点击率或用户满意度。"
            )

            with st.expander("为什么用加权和排序，而不是加权平均？", expanded=False):
                st.markdown(
                    "一开始用预测分（加权平均）排序时，Item-CF 的 HitRate@10 只有 0.03："
                    "因为加权平均要除以 Σ|sim|，只有 1 个弱邻居、恰好打了 5 分的冷门片会被归一化成 5.00，"
                    "从而挤掉“有 20 个邻居都打 4.5 分”的电影。改用加权和（Σ sim·偏差）后，"
                    "证据越多的条目分数越高，两种 CF 的 HitRate@10 都回到 0.78 左右。"
                    "这也是评分预测（E1 用加权平均）与 Top-N 推荐（E4 用加权和）必须分开评价的原因。"
                )

    section("原始文件", "可直接放进报告 / PPT")
    files = [
        ("E1 逐折", "e1_fold_results.csv"), ("E1 汇总", "e1_summary.csv"),
        ("E2 K 敏感性", "e2_k_sensitivity.csv"), ("E3 稀疏性", "e3_sparsity.csv"),
        ("E4 汇总", "e4_summary.csv"), ("E4 逐用户", "e4_per_user.csv"),
    ]
    for i in range(0, len(files), 3):
        cols = st.columns(3)
        for col, (label, path) in zip(cols, files[i:i + 3]):
            p = RESULTS / path
            if p.exists():
                col.download_button(f"{label}（{path}）", p.read_bytes(), file_name=path,
                                    use_container_width=True, key=f"dl_{path}")

    figs = sorted((RESULTS / "figures").glob("*.png"))
    if figs:
        with st.expander("实验图表（PNG，用于 PPT/报告）", expanded=False):
            for i in range(0, len(figs), 2):
                cs = st.columns(2)
                for c, f in zip(cs, figs[i:i + 2]):
                    c.image(str(f), caption=f.name, use_column_width=True)


# --------------------------------------------------------------------- 页面：概念
def page_concept():
    hero(
        "从协同过滤到深度学习推荐",
        "concept map",
        "本实验的主线始终是传统 CF。这一页只做概念过渡：把“相似度”换成“学出来的向量”。"
        "没有实测结果的模型不会被宣传成更好。",
    )

    section("三段式演进", "")
    st.markdown(
        """
<div style="display:flex; gap:14px; align-items:stretch; flex-wrap:wrap">
  <div class="kpi" style="flex:1; min-width:200px">
    <div class="label">第 1 段 · 记忆式</div>
    <div class="value" style="font-size:1rem">User-CF / Item-CF</div>
    <div class="hint">相似度直接由共现评分算出来。
    新用户没邻居 → 无法推荐；矩阵稀疏 → 相似度不可靠。</div>
  </div>
  <div style="display:flex; align-items:center; color:#4b5f80; font-size:1.4rem">→</div>
  <div class="kpi" style="flex:1; min-width:200px">
    <div class="label">第 2 段 · 表示学习</div>
    <div class="value" style="font-size:1rem">矩阵分解 MF</div>
    <div class="hint">r̂(u,i) = p_u · q_i + 偏置。
    相似度不再是离散邻居，而是向量内积。</div>
  </div>
  <div style="display:flex; align-items:center; color:#4b5f80; font-size:1.4rem">→</div>
  <div class="kpi" style="flex:1; min-width:200px">
    <div class="label">第 3 段 · 非线性交互</div>
    <div class="value" style="font-size:1rem">NCF / Embedding + MLP</div>
    <div class="hint">把 one-hot 用户/电影压成 Embedding，
    用 MLP 学非线性交互，而不是简单内积。</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        section("矩阵分解在做什么", "")
        st.markdown(
            "把 943×1682 的稀疏矩阵近似成两个矮胖矩阵的乘积：\n\n"
            "`R ≈ P (943×k) · Qᵀ (k×1682)`\n\n"
            "每个用户和每部电影都变成一个 k 维向量。**共现过的用户和电影，向量会被拉近**——"
            "这其实就是 CF 里“相似”的连续版本，只是相似度不再来自某几个人，而是来自所有交互共同训练出的几何关系。"
        )
        st.caption(
            "优点：能为「邻居不足」的用户/物品给出预测（不必先找到相似邻居）；"
            "缺点：内积仍是线性交互，而且**完全冷启动**（0 条交互的新用户/新物品）依然无解，"
            "需要内容特征或引导行为才能处理。"
        )

    with c2:
        section("NCF 在做什么", "")
        st.markdown(
            "输入用户和电影 ID → 各自查表得到 Embedding → 拼起来 → 多层感知机 → 输出预测：\n\n"
            "`ŷ(u,i) = MLP( [e_u ; e_i] )`\n\n"
            "MLP 可以表达“这个用户只在这个类型上挑剔”这类非线性关系。"
            "代价是需要更多数据、更多调参，也更难解释——**难以回答“为什么推荐给我”**，"
            "而这正是协同过滤在课堂演示里的优势。"
        )
        st.caption("NeuMF 再用一路 GMF 内积与 MLP 并联，兼顾线性与非线性。")

    section("扩展实验：Bias-MF 实测对照", "同一五折 · 同一评分预测任务 · 同一指标")
    mf_summary = read_results("mf_summary.csv")
    e1s = read_results("e1_summary.csv")
    if mf_summary is None or e1s is None:
        note("运行 <code>python scripts/run_experiments.py --only mf</code> 后，这里会显示实测对照。", warn=True)
    else:
        e1_idx = e1s.set_index("algorithm")
        m = mf_summary.iloc[0]
        rows = [
            {"模型": "用户均值基准", "MAE": f"{e1_idx.loc['baseline','mae_mean']:.4f} ± {e1_idx.loc['baseline','mae_std']:.4f}",
             "RMSE": f"{e1_idx.loc['baseline','rmse_mean']:.4f} ± {e1_idx.loc['baseline','rmse_std']:.4f}",
             "做法": "只预测该用户的训练均值"},
            {"模型": "User-CF", "MAE": f"{e1_idx.loc['user','mae_mean']:.4f} ± {e1_idx.loc['user','mae_std']:.4f}",
             "RMSE": f"{e1_idx.loc['user','rmse_mean']:.4f} ± {e1_idx.loc['user','rmse_std']:.4f}",
             "做法": "相似用户的加权偏差"},
            {"模型": "Item-CF", "MAE": f"{e1_idx.loc['item','mae_mean']:.4f} ± {e1_idx.loc['item','mae_std']:.4f}",
             "RMSE": f"{e1_idx.loc['item','rmse_mean']:.4f} ± {e1_idx.loc['item','rmse_std']:.4f}",
             "做法": "相似电影的加权偏差"},
            {"模型": "Bias-MF（扩展）", "MAE": f"{m['mae_mean']:.4f} ± {m['mae_std']:.4f}",
             "RMSE": f"{m['rmse_mean']:.4f} ± {m['rmse_std']:.4f}",
             "做法": f"μ+b_u+b_i+p_u·q_i，ALS 求解（k={int(m['k'])}，{int(m['iterations'])} 轮）"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        mf_vals = [e1_idx.loc["baseline", "rmse_mean"], e1_idx.loc["user", "rmse_mean"],
                   e1_idx.loc["item", "rmse_mean"], float(m["rmse_mean"])]
        mf_errs = [e1_idx.loc["baseline", "rmse_std"], e1_idx.loc["user", "rmse_std"],
                   e1_idx.loc["item", "rmse_std"], float(m["rmse_std"])]
        fig = go.Figure(go.Bar(
            x=["均值基准", "User-CF", "Item-CF", "Bias-MF"],
            y=mf_vals,
            error_y=dict(type="data", array=mf_errs, visible=True),
            marker_color=["#94a3b8", ACCENT, "#f59e0b", "#a855f7"],
            width=0.55,
        ))
        for label, v, e in zip(["均值基准", "User-CF", "Item-CF", "Bias-MF"], mf_vals, mf_errs):
            fig.add_annotation(x=label, y=v + e + 0.012, text=f"{v:.4f}", showarrow=False,
                               font=dict(size=11, color="#cbd6ea"))
        fig.update_layout(showlegend=False, title="五折 RMSE 对照（越低越好，误差棒为折间标准差）",
                          yaxis_range=[min(mf_vals) - 0.1, max(mf_vals) + 0.09])
        st.plotly_chart(dark(fig, 330), use_container_width=True)

        lower = "Bias-MF" if float(m["rmse_mean"]) < e1_idx["rmse_mean"].drop("baseline").min() else "传统 CF"
        st.markdown(
            f"- 实测结果：Bias-MF 的 RMSE 为 **{m['rmse_mean']:.4f}**，"
            f"User-CF {e1_idx.loc['user','rmse_mean']:.4f}、Item-CF {e1_idx.loc['item','rmse_mean']:.4f}、"
            f"均值基准 {e1_idx.loc['baseline','rmse_mean']:.4f}。这次 **{lower}** 更低。\n"
            f"- 训练成本完全不同：CF 建相似度只要几十毫秒；Bias-MF 平均每折训练 "
            f"{m['train_seconds_mean']:.1f} s（纯 CPU 的 ALS 迭代），还要调 k 与正则项。\n"
            "- 它**不参与**推荐页演示：矩阵分解很难回答“为什么是这几个人影响了预测”，"
            "而本课程的重点正是把这一步摊开。\n"
            "- 这是同一划分下的实测对照，可以放心引用；但它只覆盖评分预测任务，"
            "**没有做 Top-N 评测，也没有训练 NCF**，所以不能据此说“深度学习更好”。"
        )

    section("为什么主线只做传统 CF", "")
    st.markdown(
        "- 课程要求是讲清 User-CF / Item-CF 的基本思想，并在小数据集上跑通 Demo；"
        "传统 CF 的每一步都能摊开给人看。\n"
        "- 10 万条评分、943 用户的数据量下，CF 在 CPU 上几秒钟就能跑完五折，适合课堂现场运行。\n"
        "- 上面的 Bias-MF 已经在**同一划分、同一任务、同一指标**下做过实测，所以那一行可以引用；"
        "**但 NCF 没有做，因此不对 NCF 与 CF 的优劣下任何结论**。\n"
        "- 若继续扩展（NCF 等），也必须沿用官方五折、评分预测任务与 MAE/RMSE 指标，并明确标注为扩展实验——"
        "不要在推荐页那一侧悄悄换模型，否则前面的可解释性就白做了。"
    )

    section("选型速查", "")
    st.dataframe(pd.DataFrame([
        ["要解释“为什么推荐”", "User-CF（可指到具体的人）", "Item-CF（可指到具体的片）"],
        ["用户很少、物品很多", "邻居不足，覆盖率低", "通常更稳"],
        ["物品更新频繁", "影响较小", "需重算物品相似度"],
        ["完全没有评分的新用户", "无法推荐（找不到邻居）", "无法推荐（没有已评分的片）"],
        ["只有零星几条评分", "邻居可能不足", "通常还能工作"],
        ["本质局限", "只利用共现，不利用内容", "同左，且同质化倾向更强"],
    ], columns=["场景", "User-CF", "Item-CF"]), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------- 导航
def page_demo():
    render_demo(get_fold_model(1), get_movie_meta())


def main():
    _, _, users, items, summary = get_catalog()
    data_ok = DATA_DIR.exists() and (DATA_DIR / "u.data").exists()

    st.sidebar.markdown(
        '<div class="brand"><div class="brand-title">MovieLens Explorer</div>'
        '<div class="brand-subtitle">协同过滤课堂演示 / 100K</div></div>',
        unsafe_allow_html=True,
    )

    pages = {
        "overview": ("概览", page_overview),
        "lab": ("原理实验室", page_lab),
        "recommend": ("电影推荐", page_recommend),
        "results": ("实验结果", page_results),
        "concept": ("从 CF 到深度学习", page_concept),
    }
    labels = [v[0] for v in pages.values()]
    label_to_key = {v[0]: k for k, v in pages.items()}

    # 首次加载时支持 ?page=lab 直达完整实验；之后以侧边栏选择为准
    wanted = st.query_params.get("page", "")
    if "nav_mode" not in st.session_state:
        if wanted in pages:
            st.session_state["nav_mode"] = "完整实验"
            st.session_state["full_page"] = pages[wanted][0]
        else:
            st.session_state["nav_mode"] = "课堂演示"

    mode = st.sidebar.radio(
        "模式", ["课堂演示", "完整实验"], key="nav_mode",
        help="课堂演示：按固定案例走三步；完整实验：原有的全部页面",
    )
    if mode == "课堂演示":
        st.sidebar.caption("固定案例 · 三步看完一次协同过滤")
        st.sidebar.markdown("---")
        choice = None
    else:
        choice = label_to_key[st.sidebar.radio(
            "页面", labels, key="full_page", label_visibility="collapsed"
        )]
        st.sidebar.markdown("---")

    if data_ok:
        st.sidebar.caption(f"数据：{summary['n_ratings']:,} 条评分 · {summary['n_users']} 用户 · {summary['n_items']} 电影")
    else:
        st.sidebar.error("未找到数据，请先运行 `python scripts/download_data.py`")
    st.sidebar.caption("模型：u1.base 训练（80%），u1.test 仅用于评估")
    st.sidebar.caption("User-CF / Item-CF 均为本项目自行实现，无黑盒 predict")

    if choice is None:
        page_demo()
    else:
        pages[choice][1]()


if __name__ == "__main__":
    main()
