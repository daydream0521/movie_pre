"""课堂演示页：固定案例 + 三个连续步骤（认识用户 → 看看推荐 → 解释原因）。

设计约束：
- 固定案例由真实数据选出，运行时不改算法、不改结果；
- 每步只有一个问题、一组主视觉、一句结论；
- 主操作只有「上一步 / 下一步 / 重新演示」，不需要选用户、调 K、切算法；
- 视觉与「完整实验」共用 src/ui.py 的同一套 token 与组件，不另立设计系统；
- RMSE、覆盖率、公式、完整邻居表默认隐藏，放进「查看计算过程」。
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.collaborative_filter import ALGO_ITEM, ALGO_USER, CFModel
from src.ui import movie_card

DEMO_UID = 1          # 固定演示用户（选出规则见 docs/demo_script.md）
DEMO_K = 30           # 与完整实验一致的邻居上限，页面正文不暴露
DEMO_N = 3            # 每侧展示 3 条


# --------------------------------------------------------------------- 数据
def _iid(model: CFModel, idx: int) -> int:
    return int(model.train.item_ids[int(idx)])


def build_case(model: CFModel, meta: dict) -> dict | None:
    """按固定用户与真实模型算出三步所需的一切（不写死任何评分/推荐）。"""
    if DEMO_UID not in set(int(x) for x in model.train.user_ids):
        return None
    u = int(np.where(model.train.user_ids == DEMO_UID)[0][0])
    pop = np.asarray(model.M.sum(axis=0)).ravel()

    # 1) 认识用户：5 分电影里挑最知名的 3 部（同分按热度，保证确定性）
    five = [i for i in range(model.n_items) if model.M[u, i] and model.X[u, i] == 5]
    if len(five) < DEMO_N:
        five = [i for i in range(model.n_items) if model.M[u, i] and model.X[u, i] >= 4.5]
    five.sort(key=lambda i: (-pop[i], i))
    history = [
        {"item_id": _iid(model, i), "rating": int(model.X[u, i]),
         "title": meta[_iid(model, i)]["title"], "year": meta[_iid(model, i)]["year"],
         "genres": meta[_iid(model, i)]["genres"]}
        for i in five[:DEMO_N]
    ]

    # 类型分布：只统计“可核对的计数”，不推断偏好强弱
    genre_counts: dict[str, int] = {}
    liked = 0
    for i in range(model.n_items):
        if model.M[u, i] and model.X[u, i] >= 4:
            liked += 1
            for g in meta[_iid(model, i)]["genres_list"]:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]

    # 2) 看看推荐：两种算法的 Top-3（与推荐页同一函数、同一 K）
    rec_u = model.recommend(u, ALGO_USER, K=DEMO_K, n=DEMO_N)
    rec_i = model.recommend(u, ALGO_ITEM, K=DEMO_K, n=DEMO_N)
    set_u, set_i = set(rec_u["item_idx"]), set(rec_i["item_idx"])
    common = set_u & set_i

    def _rows(rec, side: str) -> list[dict]:
        out = []
        for k, (_, r) in enumerate(rec.iterrows()):
            iid = _iid(model, r["item_idx"])
            out.append({
                "rank": k + 1, "item_id": iid, "item_idx": int(r["item_idx"]),
                "title": meta[iid]["title"],
                "tag": "common" if r["item_idx"] in common else ("only_u" if side == "u" else "only_i"),
            })
        return out

    # 3) 解释原因：优先选两边都有邻居的共同推荐
    target = None
    best = -1
    for it in sorted(common):
        ex_u = model.explain(u, int(it), ALGO_USER, K=DEMO_K)
        ex_i = model.explain(u, int(it), ALGO_ITEM, K=DEMO_K)
        if ex_u.is_fallback or ex_i.is_fallback:
            continue
        score = min(len(ex_u.neighbors), len(ex_i.neighbors))
        if score > best:
            best, target = score, int(it)
    if target is None:                       # 兜底：没有共同项时用 User-CF 的第一名
        target = int(rec_u.iloc[0]["item_idx"])

    ex_u = model.explain(u, target, ALGO_USER, K=DEMO_K)
    ex_i = model.explain(u, target, ALGO_ITEM, K=DEMO_K)

    def _people(ex) -> list[dict]:
        rows = []
        for _, r in ex.neighbors.head(DEMO_N).iterrows():
            rows.append({"name": f"用户 {int(r['neighbor_user_id'])}", "rating": int(r["rating"]),
                         "contribution": float(r["contribution"])})
        return rows

    def _films(ex) -> list[dict]:
        rows = []
        for _, r in ex.neighbors.head(DEMO_N).iterrows():
            rows.append({"name": meta[int(r["neighbor_item_id"])]["title"],
                         "rating": int(r["user_rating"]), "contribution": float(r["contribution"])})
        return rows

    def _sum_part(ex) -> dict:
        """贡献合计与预测分的完整对账（表格只展示前几名，不能直接加总）。"""
        total = float(ex.neighbors["contribution"].sum())
        raw = float(ex.base + total)
        return {"base": float(ex.base), "total": total, "raw": raw,
                "pred": float(ex.pred), "clipped": abs(raw - float(ex.pred)) > 1e-6,
                "n_all": int(len(ex.neighbors)), "n_shown": min(DEMO_N, len(ex.neighbors))}

    in_u, in_i = target in set_u, target in set_i
    if in_u and in_i:
        rank_txt = "两种方法都把它排进了前 3"
    elif in_u:
        rank_txt = "User-CF 把它排进了前 3，Item-CF 没有"
    elif in_i:
        rank_txt = "Item-CF 把它排进了前 3，User-CF 没有"
    else:
        rank_txt = "它不在两边的 Top-3 里，这里只作为解释示例"

    return {
        "uid": DEMO_UID,
        "n_ratings": int(model.M[u].sum()),
        "n_liked": liked,
        "history": history,
        "top_genres": top_genres,
        "rec_u": _rows(rec_u, "u"),
        "rec_i": _rows(rec_i, "i"),
        "n_common": len(common),
        "target": {
            "item_id": _iid(model, target),
            "title": meta[_iid(model, target)]["title"],
            "year": meta[_iid(model, target)]["year"],
            "genres": meta[_iid(model, target)]["genres"],
            "in_u": in_u, "in_i": in_i, "rank_txt": rank_txt,
            "people": _people(ex_u), "films": _films(ex_i),
            "pred_u": float(ex_u.pred), "pred_i": float(ex_i.pred),
            "sum_u": _sum_part(ex_u), "sum_i": _sum_part(ex_i),
            "n_u": len(ex_u.neighbors), "n_i": len(ex_i.neighbors),
            "ex_u": ex_u, "ex_i": ex_i,
        },
    }


# --------------------------------------------------------------------- 渲染
# 课堂演示时隐藏侧边栏：只保留三步和两个按钮，导航放在页头的一行文字入口里
HIDE_SIDEBAR = """
<style>
section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none !important; }
</style>
"""


def _head(step: int) -> None:
    labels = ["1. 认识用户", "2. 看看推荐", "3. 解释原因"]
    tabs = "".join(
        f'<div class="dm-tab {"on" if i == step else ""}">{t}</div>' for i, t in enumerate(labels)
    )
    st.markdown(
        f'<div class="dm-head"><div><div class="dm-brand">MovieLens Explorer</div>'
        f'<div class="dm-mode">课堂演示 · 固定案例</div></div>'
        f'<div style="text-align:right"><span class="dm-pill">步骤 {step + 1}/3</span>'
        f'<div class="dm-link"><a href="?page=lab" target="_self">完整实验 →</a></div></div></div>'
        f'<div class="dm-tabs">{tabs}</div>',
        unsafe_allow_html=True,
    )


def _step1(case: dict) -> None:
    st.markdown(
        '<div class="dm-q">这位用户喜欢哪些电影？</div>'
        f'<div class="dm-sub">训练集里评过 {case["n_ratings"]} 部，这是其中 {len(case["history"])} 部满分片。</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, m in zip(cols, case["history"]):
        col.markdown(
            movie_card(m["title"], m["year"], m["genres"], m["rating"],
                       item_id=m["item_id"], cover_text=m["title"],
                       score_text=f"{m['rating']} 分"),
            unsafe_allow_html=True,
        )
    genres = "、".join(f"{g}（{c} 部）" for g, c in case["top_genres"])
    st.markdown(
        f'<div class="dm-concl">结论：他打过 4 分以上的电影共 {case["n_liked"]} 部，'
        f'其中出现最多的类型是 {genres}。</div>',
        unsafe_allow_html=True,
    )


def _step2(case: dict) -> None:
    def column(rows: list[dict], title: str, desc: str, side: str) -> str:
        items = ""
        for r in rows:
            if r["tag"] == "common":
                chip = '<span class="dm-chip same">共同</span>'
            elif r["tag"] == "only_u":
                chip = '<span class="dm-chip u">仅 User-CF</span>'
            else:
                chip = '<span class="dm-chip i">仅 Item-CF</span>'
            row_cls = "dm-row same" if r["tag"] == "common" else "dm-row"
            items += (f'<div class="{row_cls}"><span class="dm-rank">{r["rank"]:02d}</span>'
                      f'<span class="dm-name">{r["title"]}</span>{chip}</div>')
        return (f'<div><div class="dm-col-name {side}">{title}</div>'
                f'<div class="dm-col-desc">{desc}</div>{items}</div>')

    st.markdown(
        '<div class="dm-q">两种算法分别推荐了什么？</div>'
        '<div class="dm-sub">同一份历史评分，两种“相似”的定义。</div>',
        unsafe_allow_html=True,
    )
    left = column(case["rec_u"], "User-CF", "找口味相似的人", "u")
    right = column(case["rec_i"], "Item-CF", "找评分模式相似的电影", "i")
    st.markdown(f'<div class="dm-cols">{left}{right}</div>', unsafe_allow_html=True)
    same = case["n_common"]
    tail = ("其余推荐完全不同——两种方法看的不是同一件事。" if same < 3
            else "两种方法在这位用户身上给出了相同的前三名。")
    st.markdown(
        f'<div class="dm-concl">结论：3 个位置里有 {same} 部重合，{tail}</div>',
        unsafe_allow_html=True,
    )


def _step3(case: dict) -> None:
    t = case["target"]
    st.markdown(
        f'<div class="dm-q">为什么推荐《{t["title"]}》？</div>'
        '<div class="dm-sub">同一部电影，两套证据。</div>',
        unsafe_allow_html=True,
    )
    year = int(t["year"]) if t["year"] == t["year"] else "—"
    st.markdown(
        f'<div class="dm-film"><div>'
        f'<div class="t">{t["title"]} ({year})</div>'
        f'<div class="g">{t["genres"]}　·　{t["rank_txt"]}</div></div></div>',
        unsafe_allow_html=True,
    )

    def rows_html(rows: list[dict]) -> str:
        top = max((abs(r["contribution"]) for r in rows), default=1.0) or 1.0
        out = ""
        for k, r in enumerate(rows, 1):
            sign = "pos" if r["contribution"] >= 0 else "neg"
            width = max(6, round(abs(r["contribution"]) / top * 100))
            out += (f'<div class="dm-row"><span class="dm-rank">{k:02d}</span>'
                    f'<span class="dm-name">{r["name"]}</span>'
                    f'<span class="dm-meta">{r["rating"]} 分</span>'
                    f'<span class="dm-bar"><i class="{"neg" if r["contribution"] < 0 else ""}" '
                    f'style="width:{width}%"></i></span>'
                    f'<span class="dm-chip {sign}">{r["contribution"]:+.3f}</span></div>')
        return out

    left = (f'<div><div class="dm-col-name u">User-CF：人的证据</div>'
            f'<div class="dm-col-desc">这些相似用户喜欢它，所以推荐给你</div>'
            f'{rows_html(t["people"])}</div>')
    right = (f'<div><div class="dm-col-name i">Item-CF：电影的证据</div>'
             f'<div class="dm-col-desc">这些你看过的电影和它相似，评分都很高</div>'
             f'{rows_html(t["films"])}</div>')
    st.markdown(f'<div class="dm-cols">{left}{right}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dm-concl">结论：左边看“和你像的人”，右边看“和它像的片”，'
        '两条证据指向同一个推荐；末尾 +/− 是该邻居对预测分的影响。</div>',
        unsafe_allow_html=True,
    )

    with st.expander("查看计算过程", expanded=False):
        su, si = t["sum_u"], t["sum_i"]

        def check(part: dict) -> str:
            line = (f"用户均值 {part['base']:.3f} + 全部 {part['n_all']} 个邻居的贡献合计 "
                    f"{part['total']:+.3f} = {part['raw']:.3f}")
            if part["clipped"]:
                return line + f"（超出 1–5，被裁剪为 {part['pred']:.2f}）"
            return line + f"，与预测分 {part['pred']:.2f} 一致"

        st.markdown(
            "**预测公式（两种算法只在“邻居是谁”上不同）**\n\n"
            "`预测 = 该用户均值 + Σ 相似度 × (邻居评分 − 邻居均值) / Σ|相似度|`\n\n"
            f"- 用户 ID：{case['uid']}；训练集评分 {case['n_ratings']} 条；"
            f"邻居上限 K = {DEMO_K}（与完整实验一致，正文不展示这个参数）\n"
            f"- 表格只列出贡献最大的 {su['n_shown']} 个邻居，"
            f"**不能把这几行直接相加得到预测分**；完整对账如下：\n"
            f"  - User-CF（{t['n_u']} 个邻居）：{check(su)}\n"
            f"  - Item-CF（{t['n_i']} 个邻居）：{check(si)}\n"
            f"- 单元格里的 +0.xxx / −0.xxx 是单个邻居的贡献（已除以 Σ|相似度|），"
            "正贡献抬高预测、负贡献拉低预测。"
        )
        st.markdown("**User-CF 的完整邻居表（前 5）**")
        st.dataframe(t["ex_u"].neighbors.head(5), hide_index=True, use_container_width=True)
        st.markdown("**Item-CF 的完整邻居表（前 5）**")
        st.dataframe(t["ex_i"].neighbors.head(5), hide_index=True, use_container_width=True)


def render_demo(model: CFModel, meta: dict) -> None:
    """课堂演示页入口：三步 + 两个按钮（样式全部来自 src/ui.py）。"""
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    case = build_case(model, meta)
    if case is None:
        st.warning("演示案例不可用：请先运行 `python scripts/download_data.py`。")
        return

    step = int(st.session_state.get("demo_step", 0))
    step = max(0, min(2, step))
    st.session_state["demo_step"] = step

    _head(step)

    if step == 0:
        _step1(case)
    elif step == 1:
        _step2(case)
    else:
        _step3(case)

    st.markdown('<hr class="dm-hr">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c1:
        if st.button("← 上一步", key="demo_prev", use_container_width=True, disabled=(step == 0)):
            st.session_state["demo_step"] = step - 1
            st.rerun()
    with c3:
        if step < 2:
            if st.button("下一步 →", key="demo_next", type="primary", use_container_width=True):
                st.session_state["demo_step"] = step + 1
                st.rerun()
        else:
            if st.button("重新演示", key="demo_restart", type="primary", use_container_width=True):
                st.session_state["demo_step"] = 0
                st.rerun()
