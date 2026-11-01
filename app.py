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
from src.tutorial import (
    DEFAULT_K, DEFAULT_TARGET_ITEM, DEFAULT_TARGET_USER, TUTORIAL_MOVIES, TUTORIAL_USERS,
    hand_calculation, movie_label, rating_frame, steps as tutorial_steps,
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
    initial_sidebar_state="expanded",
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
        font=dict(color="#c9d5ea", family="Inter, Microsoft YaHei", size=12),
        margin=dict(l=8, r=8, t=36, b=8),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
    )
    if fig.layout.title.text:
        fig.update_layout(title=dict(font=dict(size=13, color="#e6eefc")))
    else:
        fig.update_layout(margin=dict(l=8, r=8, t=14, b=8))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# --------------------------------------------------------------------- 页面：概览
def page_overview():
    ratings, movies, users, items, summary = get_catalog()
    hero(
        "MovieLens Explorer",
        "recommender systems · user-cf & item-cf",
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

    steps = tutorial_steps(algo, user_idx, item_idx, K)
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
        st.markdown(matrix_table(data["matrix"], data["user"], data["item"]), unsafe_allow_html=True)
        st.caption("行 = 用户，列 = 电影；“—” 表示没有评分（缺失，不是 0 分）；蓝色行是目标用户，紫色列是目标电影。")

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
        st.markdown(hand_calculation(algo, user_idx, item_idx, K))

    with st.expander("老师可能会问", expanded=False):
        st.markdown(
            """
**1. 空白格子能当成 0 分吗？**
不能。0 分会把“没看过”误判成“很讨厌”，用户均值被拉低，相似度也会失真。计算相似度时只统计双方共同评过分的电影。

**2. 为什么要减去用户均值（中心化）？**
有人习惯全打 4–5 分，有人只打 2–3 分。中心化后比较的是“相对于他自己，这部片算好还是差”，相似度才反映口味而不是打分习惯。

**3. 相似度为什么会出现负数？**
中心化向量的夹角超过 90°，说明这两人口味相反。负相似度在预测时会反向外推，通常直接不使用——所以本实现只取正相似邻居。

**4. 共同评分为 1 部电影时，相似度能信吗？**
不能。一部片的巧合不足以说明口味，真实实验中要求共同评分 ≥ 3，并用 n/(n+10) 收缩相似度（本页小矩阵为便于手算关闭了这两项）。

**5. User-CF 和 Item-CF 的差别到底在哪？**
User-CF 找“人”（谁给我参考分），Item-CF 找“物”（我喜欢的片还像哪部）。前者依赖用户数量，后者依赖物品数量；用户口味漂移时 Item-CF 通常更稳。
            """
        )


# --------------------------------------------------------------------- 页面：电影推荐
def page_recommend():
    ratings, movies, users, items, summary = get_catalog()
    meta = get_movie_meta()
    model = get_fold_model(1)
    train_counts = np.asarray(model.M.sum(axis=1)).ravel()

    hero(
        "电影推荐体验",
        "real data · u1.base 训练",
        "从 943 名真实用户中选一位，用训练集里他/她的历史评分生成 Top-10，"
        "并展开任一条推荐的加权依据。测试集评分不参与推荐。",
    )

    c1, c2, c3, c4 = st.columns([1.9, 1.1, 1.0, 1.0])
    default_user = st.session_state.get("rec_user", int(users[0]))
    uid = c1.selectbox(
        "选择一个用户", [int(u) for u in users],
        index=list(users).index(default_user),
        format_func=lambda u: f"用户 {u} · u1.base 中 {int(train_counts[int(np.where(model.train.user_ids == u)[0][0])])} 条评分",
        key="rec_user_select",
    )
    st.session_state["rec_user"] = uid
    algo_label = c2.radio("算法", ["User-CF", "Item-CF"], horizontal=True, key="rec_algo")
    algo = ALGO_USER if algo_label == "User-CF" else ALGO_ITEM
    K = c3.select_slider("邻居数 K", options=[5, 10, 20, 30, 50], value=30, key="rec_k")
    topn = c4.selectbox("推荐数量", [5, 10, 20], index=1, key="rec_n")

    if st.button("换一个用户", key="rec_random"):
        st.session_state["rec_user"] = random.choice([int(u) for u in users])
        st.rerun()

    u_idx = int(np.where(model.train.user_ids == uid)[0][0])
    n_train_ratings = int(model.M[u_idx].sum())
    stats = model.neighbor_stats(u_idx, algo)
    rec = model.recommend(u_idx, algo, K=K, n=topn)

    kpi_row([
        ("该用户训练集评分", f"{n_train_ratings}", "u1.base 中出现过的记录"),
        ("全库矩阵密度", f"{summary['density']*100:.2f}%", f"{model.n_users}×{model.n_items} 个格子"),
        ("可用正相似" + ("用户" if algo == ALGO_USER else "电影"), f"{stats['positive_neighbors']}", "相似度 > 0 的候选"),
        ("候选电影", f"{model.n_items - n_train_ratings}", "已排除他/她看过的片"),
        ("回退条数", f"{int(rec['is_fallback'].sum())}/{len(rec)}", "无邻居时取用户均值"),
    ])

    left, right = st.columns([1.05, 1])

    with left:
        section("他/她给过的高分电影", "训练集中评分 = 5 的代表作")
        seen = model.train.ratings_of(u_idx)
        seen = seen[seen["rating"] >= 4.5].merge(
            pd.DataFrame({"item_idx": np.arange(model.n_items)}), on="item_idx"
        )
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
            st.caption("该用户在训练集中没有 5 分记录。")

        section("评分偏好", "最近 60 条训练评分")
        hist = np.bincount(model.X[u_idx][model.M[u_idx]].astype(int), minlength=6)[1:6]
        fig = go.Figure(go.Bar(x=[f"{i} 分" for i in range(1, 6)], y=hist,
                               marker_color=[ACCENT, ACCENT, "#60a5fa", GOOD, GOOD]))
        fig.update_layout(showlegend=False, title=f"用户 {uid} 的评分分布（均值 {model.mu[u_idx]:.2f}）")
        st.plotly_chart(dark(fig, 250), use_container_width=True)

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
            fig = go.Figure(go.Bar(x=gser.values, y=gser.index, orientation="h",
                                   marker_color=ACCENT_2))
            fig.update_layout(showlegend=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(dark(fig, 250), use_container_width=True)
        else:
            st.caption("该用户没有 4 分以上的评分。")

        section("这条推荐是怎么算出来的", "")
        note(
            f"当前算法：<b>{algo_label}</b>；K = {K}。"
            + ("邻居是“同样看过这部片、且与目标用户相似度为正值”的人。"
               if algo == ALGO_USER else "邻居是“目标用户看过、且与这部片相似度为正值”的电影。")
            + " 回退为用户均值的条目会单独标注。"
        )

    section(f"Top-{topn} 推荐", f"{algo_label} · K = {K} · 已排除训练集看过的 {n_train_ratings} 部")
    rec_cols = st.columns(5)
    for i, (_, r) in enumerate(rec.iterrows()):
        item_id = int(model.train.item_ids[int(r["item_idx"])])
        m = meta[item_id]
        nbr_txt = f"{int(r['n_neighbors'])} 个邻居支撑" if not r["is_fallback"] else "无邻居"
        rec_cols[i % 5].markdown(
            movie_card(m["title"], m["year"], m["genres"], float(r["score"]),
                       meta=f"预测 {r['score']:.2f} · {nbr_txt}", rank=i + 1,
                       fallback=bool(r["is_fallback"]), item_id=item_id),
            unsafe_allow_html=True,
        )
    capped = int((rec["score"] >= 4.995).sum())
    if capped:
        note(
            f"有 {capped} 条预测触顶 5.00：这些目标电影的可用邻居都给了高分，加权后超过上限被裁剪到 5 分。"
            "看“邻居支撑”列——只靠 1 个邻居的条目证据最弱，这正是 K 与共同评分门槛要解决的问题。"
        )

    section("查看单条推荐的依据", "选一部，看加权过程")
    options = [
        f"#{i+1}  {meta[int(model.train.item_ids[int(r['item_idx'])])]['title']}（预测 {r['score']:.2f}）"
        for i, (_, r) in enumerate(rec.iterrows())
    ]
    pick = st.selectbox("推荐条目", options, index=0, key="rec_pick")
    row = rec.iloc[options.index(pick)]
    item_idx = int(row["item_idx"])
    ex = model.explain(u_idx, item_idx, algo, K=K)
    item_id = int(model.train.item_ids[item_idx])
    m = meta[item_id]

    a, b = st.columns([0.8, 1.6])
    with a:
        st.markdown(
            f'<div class="kpi"><div class="label">{m["genres"]}</div>'
            f'<div class="value" style="font-size:1.15rem">{m["title"]} ({int(m["year"]) if pd.notna(m["year"]) else "—"})</div>'
            f'<div class="hint">mean(u) = {ex.base:.3f}　Σ|sim| = {ex.sim_sum:.3f}　'
            f'{"均值回退" if ex.is_fallback else "协同过滤"}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(ex.formula)

    with b:
        if ex.neighbors.empty:
            note("这个目标没有可用的正相似邻居，预测退化为该用户的训练均值，不能当作协同过滤的结果。", warn=True)
        else:
            nb = ex.neighbors.copy()
            if algo == ALGO_USER:
                nb["来源"] = ["用户 " + str(int(x)) for x in nb["neighbor_user_id"]]
                nb["TA 的评分"] = nb["rating"]
                nb["相似度加权贡献"] = nb["contribution"]
                nb = nb[["来源", "similarity", "TA 的评分", "相似度加权贡献"]]
                nb.columns = ["来源（相似用户）", "相似度", "TA 给这部片的评分", "对预测的贡献"]
            else:
                nb["来源"] = [
                    meta[int(model.train.item_ids[int(x)])]["title"] for x in nb["neighbor_item_id"]
                ]
                nb["相似度加权贡献"] = nb["contribution"]
                nb = nb[["来源", "similarity", "user_rating", "相似度加权贡献"]]
                nb.columns = ["来源（相似电影）", "相似度", "你给它的评分", "对预测的贡献"]
            st.dataframe(nb.style.format({
                "相似度": "{:.4f}", "对预测的贡献": "{:+.4f}",
                "TA 给这部片的评分": "{:.0f}", "你给它的评分": "{:.0f}",
            }), hide_index=True, use_container_width=True)
            cnt = nb.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=cnt["对预测的贡献"], y=cnt["来源（" + ("相似用户" if algo == ALGO_USER else "相似电影") + "）"],
                orientation="h", marker_color=[GOOD if v >= 0 else "#ef4444" for v in cnt["对预测的贡献"]],
            ))
            fig.update_layout(showlegend=False, title="各邻居对最终预测分的贡献（分）")
            st.plotly_chart(dark(fig, max(220, 26 * len(cnt))), use_container_width=True)

    with st.expander("为什么同一条推荐在不同 K 下会变化？", expanded=False):
        st.markdown(
            "K 限制的是**预测时实际参与的正相似邻居数量**。K 小 → 只用最像的几个，方差大但更贴合；"
            "K 大 → 平均更稳，但会把“没那么像”的人也算进来，可能向人群均值靠拢。"
            "每张卡片上的“N 个邻居支撑”就是这个条目实际用到的邻居数。"
        )


# --------------------------------------------------------------------- 页面：实验结果
def page_results():
    hero(
        "实验结果",
        "e1 · e2 · e3",
        "全部指标由 scripts/run_experiments.py 在官方五折上真实跑出，页面只读取 results/*.csv。"
        "图表不写死结论，只标注怎么读。",
    )
    e1 = read_results("e1_fold_results.csv")
    s1 = read_results("e1_summary.csv")
    e2 = read_results("e2_k_sensitivity.csv")
    e3 = read_results("e3_sparsity.csv")

    if e1 is None:
        note("还没有实验输出。请在项目根目录运行：<code>python scripts/run_experiments.py</code>", warn=True)
        return

    tab1, tab2, tab3 = st.tabs(["E1 五折算法比较", "E2 邻居数 K", "E3 训练记录减少"])

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
            for a, color in zip(order, ["#94a3b8", ACCENT, "#f59e0b"]):
                fig.add_trace(go.Bar(name=names[a], x=[names[a]], y=[idx.loc[a, "coverage_mean"] * 100],
                                     marker_color=color, text=[f"{idx.loc[a, 'coverage_mean']*100:.2f}%"],
                                     textposition="outside"))
            fig.update_layout(showlegend=False, title="预测覆盖率（有 ≥1 个有效邻居的测试评分占比）",
                              yaxis_title="%", yaxis_range=[0, 108])
            st.plotly_chart(dark(fig, 320), use_container_width=True)
        st.caption("基准模型覆盖率恒为 100%：它不需要邻居，只用用户均值。")

        section("逐折明细", "5 个官方 fold")
        show = e1[["fold", "algorithm", "mae", "rmse", "coverage", "fallback_rate", "mean_neighbors",
                   "build_seconds", "predict_seconds"]].copy()
        show["algorithm"] = show["algorithm"].map(names)
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

    section("原始文件", "可直接放进报告 / PPT")
    cols = st.columns(4)
    for col, (name, path) in zip(cols, [
        ("E1 逐折", "e1_fold_results.csv"), ("E1 汇总", "e1_summary.csv"),
        ("E2 K 敏感性", "e2_k_sensitivity.csv"), ("E3 稀疏性", "e3_sparsity.csv"),
    ]):
        p = RESULTS / path
        if p.exists():
            col.download_button(path, p.read_bytes(), file_name=path, use_container_width=True)

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
        st.caption("优点：能泛化到邻居为空的用户/物品；缺点：内积仍是线性交互，表达能力有限。")

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

    section("为什么本实验只做传统 CF", "")
    st.markdown(
        "- 课程要求是讲清 User-CF / Item-CF 的基本思想，并在小数据集上跑通 Demo；"
        "传统 CF 的每一步都能摊开给人看。\n"
        "- 10 万条评分、943 用户的数据量下，CF 在 CPU 上几秒钟就能跑完五折，适合课堂现场运行。\n"
        "- 深度学习模型在这个数据规模上的收益很小且不稳定，且需要 GPU/调参预算；"
        "**在没有实测对照的情况下，不应声称 NCF 优于 CF**。\n"
        "- 若后续要做扩展实验，必须在同一划分（官方五折）、同一任务（评分预测）、同一指标（MAE/RMSE）下比较，"
        "并把 NCF 明确定位为“扩展实验”。"
    )

    section("选型速查", "")
    st.dataframe(pd.DataFrame([
        ["要解释“为什么推荐”", "User-CF（可指到具体的人）", "Item-CF（可指到具体的片）"],
        ["用户很少、物品很多", "邻居不足，覆盖率低", "通常更稳"],
        ["物品更新频繁", "影响较小", "需重算物品相似度"],
        ["冷启动用户", "无法推荐（无邻居）", "可用少量已看片推断"],
        ["本质局限", "只利用共现，不利用内容", "同左，且同质化倾向更强"],
    ], columns=["场景", "User-CF", "Item-CF"]), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------- 导航
def main():
    _, _, users, items, summary = get_catalog()
    data_ok = DATA_DIR.exists() and (DATA_DIR / "u.data").exists()

    st.sidebar.markdown(
        '<div style="font-size:1.15rem;font-weight:700;color:#f2f6ff;padding:4px 0 2px">🎬 MovieLens Explorer</div>'
        '<div style="color:#7f91b0;font-size:0.76rem;margin-bottom:16px">协同过滤课堂演示 · MovieLens 100K</div>',
        unsafe_allow_html=True,
    )
    pages = {
        "overview": ("概览", page_overview),
        "lab": ("原理实验室", page_lab),
        "recommend": ("电影推荐", page_recommend),
        "results": ("实验结果", page_results),
        "concept": ("从 CF 到深度学习", page_concept),
    }
    keys = list(pages)
    labels = [pages[k][0] for k in keys]
    # 支持 ?page=lab 直达某个页面（课堂演示时可直接收藏）
    wanted = st.query_params.get("page", "")
    default_index = keys.index(wanted) if wanted in pages else 0
    picked = st.sidebar.radio("导航", labels, index=default_index, label_visibility="collapsed")
    choice = keys[labels.index(picked)]
    st.sidebar.markdown("---")
    if data_ok:
        st.sidebar.caption(f"数据：{summary['n_ratings']:,} 条评分 · {summary['n_users']} 用户 · {summary['n_items']} 电影")
    else:
        st.sidebar.error("未找到数据，请先运行 `python scripts/download_data.py`")
    st.sidebar.caption("模型：u1.base 训练（80%），u1.test 仅用于评估")
    st.sidebar.caption("User-CF / Item-CF 均为本项目自行实现，无黑盒 predict")

    pages[choice][1]()


if __name__ == "__main__":
    main()
