"""课堂演示页：固定案例 + 三个连续步骤（认识用户 → 看看推荐 → 解释原因）。

设计约束（来自课堂可用性要求）：
- 固定案例由真实数据选出，运行时不改算法、不改结果；
- 每步只有一个问题、一组主视觉、一句结论；
- 主操作只有「上一步 / 下一步 / 重新演示」，不需要选用户、调 K、切算法；
- RMSE、覆盖率、公式、完整邻居表默认隐藏，放进「查看计算过程」。
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.collaborative_filter import ALGO_ITEM, ALGO_USER, CFModel

DEMO_UID = 1          # 固定演示用户（选出规则见 docs/demo_script.md）
DEMO_K = 30           # 与完整实验一致的预测邻居上限，页面上不暴露
DEMO_N = 3            # 每侧展示 3 条

CSS = """
<style>
.stApp { background:#f3f4f6; }
.block-container {
  background:#ffffff; border:1px solid #e5e7eb; border-radius:16px;
  padding:22px 30px 18px; margin:18px auto 24px; max-width:1080px;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
}
.block-container p, .block-container li { color:#374151; }
h1,h2,h3,h4 { color:#111827; }
.dm-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }
.dm-brand { font-size:1.38rem; font-weight:750; color:#111827; letter-spacing:-.01em; }
.dm-mode { color:#6b7280; font-size:.82rem; margin-top:3px; }
.dm-step-pill { background:#f3f4f6; color:#4b5563; border-radius:999px;
  padding:5px 12px; font-size:.78rem; white-space:nowrap; }
.dm-tabs { display:flex; gap:14px; margin:16px 0 14px; }
.dm-tab { flex:1; text-align:center; padding:9px 8px; border-radius:9px;
  color:#6b7280; font-size:.88rem; }
.dm-tab.on { background:#e8f0fe; color:#1d4ed8; font-weight:650; }
.dm-q { font-size:1.32rem; font-weight:750; color:#111827; margin:2px 0 4px; }
.dm-sub { color:#6b7280; font-size:.86rem; margin-bottom:14px; }
.dm-cards { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.dm-card { border:1px solid #eef1f5; border-radius:12px; overflow:hidden; background:#fff; }
.dm-cover { height:132px; background:#eef3f9; display:flex; flex-direction:column;
  align-items:center; justify-content:center; color:#94a3b8; gap:6px; }
.dm-cover .ico { font-size:1.5rem; opacity:.65; }
.dm-cover .yr { font-size:.72rem; letter-spacing:.04em; }
.dm-body { padding:10px 12px 14px; }
.dm-title { font-size:.94rem; font-weight:650; color:#111827; line-height:1.35;
  min-height:2.6em; overflow-wrap:anywhere; }
.dm-rate { color:#b45309; font-size:.82rem; margin-top:6px; }
.dm-cols { display:grid; grid-template-columns:1fr 1fr; gap:26px; }
.dm-col-name { font-size:1rem; font-weight:750; color:#111827; }
.dm-col-desc { color:#6b7280; font-size:.8rem; margin:2px 0 10px; }
.dm-row { display:flex; align-items:center; gap:10px; padding:8px 0;
  border-bottom:1px solid #f1f3f6; }
.dm-row:last-child { border-bottom:none; }
.dm-rank { color:#9ca3af; font-size:.8rem; width:18px; font-variant-numeric:tabular-nums; }
.dm-name { flex:1; font-size:.9rem; color:#111827; overflow-wrap:anywhere; }
.dm-chip { font-size:.7rem; padding:2px 7px; border-radius:999px; white-space:nowrap; }
.dm-chip.c { background:#ecfdf5; color:#047857; }
.dm-chip.u { background:#fef3c7; color:#b45309; }
.dm-chip.i { background:#eff6ff; color:#1d4ed8; }
.dm-meta { color:#9ca3af; font-size:.74rem; }
.dm-film { background:#f8fafc; border:1px solid #eef1f5; border-radius:12px;
  padding:14px 16px; display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.dm-film .t { font-size:1rem; font-weight:700; color:#111827; }
.dm-film .g { color:#6b7280; font-size:.78rem; margin-top:2px; }
.dm-concl { margin-top:14px; padding:11px 14px; background:#fafaf9;
  border-left:3px solid #b45309; border-radius:0 8px 8px 0;
  color:#44403c; font-size:.88rem; line-height:1.55; }
.dm-hr { border:none; border-top:1px solid #eef1f5; margin:14px 0 10px; }
div[data-testid="stButton"] button { border-radius:999px; min-height:40px; font-size:.9rem; }
div[data-testid="stButton"] button[kind="primary"] { background:#a9502b; border:1px solid #a9502b;
  color:#ffffff; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#8f4223; border-color:#8f4223;
  color:#ffffff; }
div[data-testid="stButton"] button[kind="secondary"] { background:#ffffff; border:1px solid #d1d5db;
  color:#4b5563; }
div[data-testid="stButton"] button:disabled { background:#f9fafb; color:#d1d5db; border-color:#e5e7eb; }
[data-testid="stExpander"] { border:1px solid #eef1f5; border-radius:10px; }
[data-testid="stExpander"] summary { color:#6b7280; font-size:.82rem; }
[data-testid="stExpander"] p, [data-testid="stExpander"] li,
[data-testid="stExpander"] td, [data-testid="stExpander"] th { color:#374151; }
[data-testid="stDataFrame"] { border:1px solid #eef1f5; border-radius:8px; }
</style>
"""


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
         "title": meta[_iid(model, i)]["title"], "year": meta[_iid(model, i)]["year"]}
        for i in five[:DEMO_N]
    ]

    # 高分电影的类型偏好（一句话结论用）
    counts: dict[str, int] = {}
    for i in range(model.n_items):
        if model.M[u, i] and model.X[u, i] >= 4:
            for g in meta[_iid(model, i)]["genres_list"]:
                counts[g] = counts.get(g, 0) + 1
    top_genres = [g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

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
    if target is None:
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

    return {
        "uid": DEMO_UID,
        "n_ratings": int(model.M[u].sum()),
        "n_five": len(five),
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
            "in_u": target in set_u, "in_i": target in set_i,
            "people": _people(ex_u), "films": _films(ex_i),
            "pred_u": float(ex_u.pred), "pred_i": float(ex_i.pred),
            "n_u": len(ex_u.neighbors), "n_i": len(ex_i.neighbors),
            "ex_u": ex_u, "ex_i": ex_i,
        },
    }


# --------------------------------------------------------------------- 渲染
def _head(step: int) -> None:
    labels = ["1. 认识用户", "2. 看看推荐", "3. 解释原因"]
    tabs = "".join(
        f'<div class="dm-tab {"on" if i == step else ""}">{t}</div>' for i, t in enumerate(labels)
    )
    st.markdown(
        f'<div class="dm-head"><div><div class="dm-brand">MovieLens Explorer</div>'
        f'<div class="dm-mode">课堂演示 · 固定案例</div></div>'
        f'<div class="dm-step-pill">步骤 {step + 1}/3</div></div>'
        f'<div class="dm-tabs">{tabs}</div>',
        unsafe_allow_html=True,
    )


def _step1(case: dict) -> None:
    st.markdown(
        '<div class="dm-q">这位用户喜欢哪些电影？</div>'
        f'<div class="dm-sub">他在训练集里评过 {case["n_ratings"]} 部电影，'
        f'下面是其中 {len(case["history"])} 部打了满分的。</div>',
        unsafe_allow_html=True,
    )
    cards = "".join(
        f'<div class="dm-card"><div class="dm-cover"><div class="ico">🎬</div>'
        f'<div class="yr">{int(m["year"]) if m["year"] == m["year"] else ""}</div></div>'
        f'<div class="dm-body"><div class="dm-title">{m["title"]}</div>'
        f'<div class="dm-rate">{"★" * m["rating"]} {m["rating"]} 分</div></div></div>'
        for m in case["history"]
    )
    st.markdown(f'<div class="dm-cards">{cards}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dm-concl">结论：他偏爱 {(" / ".join(case["top_genres"]))} 这类电影，'
        f'接下来看看系统会给他推什么。</div>',
        unsafe_allow_html=True,
    )


def _step2(case: dict) -> None:
    def column(rows: list[dict], title: str, desc: str, chip_cls: str, chip_txt: str) -> str:
        items = ""
        for r in rows:
            chip = ('<span class="dm-chip c">共同</span>' if r["tag"] == "common"
                    else f'<span class="dm-chip {chip_cls}">{chip_txt}</span>')
            items += (f'<div class="dm-row"><span class="dm-rank">{r["rank"]:02d}</span>'
                      f'<span class="dm-name">{r["title"]}</span>{chip}</div>')
        return (f'<div><div class="dm-col-name">{title}</div>'
                f'<div class="dm-col-desc">{desc}</div>{items}</div>')

    st.markdown(
        '<div class="dm-q">两种算法分别推荐了什么？</div>'
        '<div class="dm-sub">同一个用户、同一份历史评分，只有“相似”的定义不同。</div>',
        unsafe_allow_html=True,
    )
    left = column(case["rec_u"], "User-CF", "找口味相似的人", "u", "仅 User-CF")
    right = column(case["rec_i"], "Item-CF", "找评分模式相似的电影", "i", "仅 Item-CF")
    st.markdown(f'<div class="dm-cols">{left}{right}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dm-concl">结论：3 个位置里只有 {case["n_common"]} 部重合，'
        '其余推荐完全不同——两种方法看的不是同一件事。</div>',
        unsafe_allow_html=True,
    )


def _step3(case: dict) -> None:
    t = case["target"]
    st.markdown(
        f'<div class="dm-q">为什么推荐《{t["title"]}》？</div>'
        '<div class="dm-sub">同一部电影，两种算法各自给出的证据。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="dm-film"><div style="font-size:1.4rem">🎬</div><div>'
        f'<div class="t">{t["title"]} ({int(t["year"]) if t["year"] == t["year"] else "—"})</div>'
        f'<div class="g">{t["genres"]}　·　两种方法都把它排进了前 3</div></div></div>',
        unsafe_allow_html=True,
    )

    def rows_html(rows: list[dict]) -> str:
        out = ""
        for k, r in enumerate(rows, 1):
            out += (f'<div class="dm-row"><span class="dm-rank">{k:02d}</span>'
                    f'<span class="dm-name">{r["name"]}</span>'
                    f'<span class="dm-meta">{r["rating"]} 分</span>'
                    f'<span class="dm-chip c">+{r["contribution"]:.3f}</span></div>')
        return out

    left = (f'<div><div class="dm-col-name">User-CF：人的证据</div>'
            f'<div class="dm-col-desc">这些相似用户喜欢它，所以推荐给你</div>'
            f'{rows_html(t["people"])}</div>')
    right = (f'<div><div class="dm-col-name">Item-CF：电影的证据</div>'
             f'<div class="dm-col-desc">这些你看过的电影和它相似，评分都很高</div>'
             f'{rows_html(t["films"])}</div>')
    st.markdown(f'<div class="dm-cols">{left}{right}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dm-concl">结论：左边看“和你像的人”，右边看“和它像的片”，'
        '两条证据链指向同一个推荐。</div>',
        unsafe_allow_html=True,
    )

    with st.expander("查看计算过程", expanded=False):
        st.markdown(
            f"**预测公式（两种算法只在“邻居是谁”上不同）**\n\n"
            f"`预测 = 该用户均值 + Σ 相似度 × (邻居评分 − 邻居均值) / Σ|相似度|`\n\n"
            f"- 用户 ID：{case['uid']}；训练集评分 {case['n_ratings']} 条；邻居上限 K = {DEMO_K}"
            f"（与完整实验一致，页面上不暴露这个参数）\n"
            f"- 用户均值：{t['ex_u'].base:.3f}\n"
            f"- 这部片：User-CF 用 {t['n_u']} 个相似用户，预测分 **{t['pred_u']:.2f}**；"
            f"Item-CF 用 {t['n_i']} 部相似电影，预测分 **{t['pred_i']:.2f}**\n"
            f"- 右表的“+0.xxx”是每个邻居对预测分的实际贡献（已除以 Σ|相似度|），"
            f"加总后即上面的预测分。"
        )
        st.markdown("**User-CF 的完整邻居表（前 5）**")
        st.dataframe(t["ex_u"].neighbors.head(5), hide_index=True, use_container_width=True)
        st.markdown("**Item-CF 的完整邻居表（前 5）**")
        st.dataframe(t["ex_i"].neighbors.head(5), hide_index=True, use_container_width=True)


def render_demo(model: CFModel, meta: dict) -> None:
    """课堂演示页入口：三步 + 两个按钮。"""
    st.markdown(CSS, unsafe_allow_html=True)
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
