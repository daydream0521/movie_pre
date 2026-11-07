"""界面层：配色、卡片、图表等可复用渲染片段（全部离线，不依赖海报网络图片）。"""

from __future__ import annotations

import hashlib

import pandas as pd

ACCENT = "#efb85b"
ACCENT_2 = "#67c6c0"
GOOD = "#88d3b2"
WARN = "#efb85b"

GRADIENTS = [
    ("#30475a", "#30475a"),
    ("#44504f", "#44504f"),
    ("#28514f", "#28514f"),
    ("#594631", "#594631"),
    ("#57404a", "#57404a"),
    ("#3e485a", "#3e485a"),
]

CSS = """
<style>
:root {
 --canvas:#141c24; --surface:#1c2833; --surface-2:#23313e; --line:#344451;
 --line-soft:rgba(255,255,255,.07);
 --ink:#edf1f2; --muted:#a7b6c1; --accent:#efb85b; --compare:#67c6c0;
 --good:#88d3b2; --bad:#e07a7a;
 --t:200ms; --ease:cubic-bezier(.2,.8,.2,1);
}
html, body, .stApp, input, button, textarea, select {
 font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
}
.stApp { background:var(--canvas); color:var(--ink); }
section[data-testid="stSidebar"] { background:#101820; border-right:1px solid var(--line); }
#MainMenu, footer, header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
.block-container { padding-top:2.2rem; padding-bottom:1.8rem; max-width:1280px; }
h1,h2,h3,h4 { color:var(--ink); }
p,li { line-height:1.7; }
a { color:var(--accent); }
.hero { padding:8px 0 28px 24px; margin-bottom:24px; border-left:4px solid var(--accent); }
.hero .kicker { color:var(--accent); font-size:.9rem; font-weight:600; margin-bottom:10px; }
.hero h1 { font-size:clamp(1.8rem,3vw,2.65rem); line-height:1.2; margin:0 0 14px; font-weight:700; letter-spacing:-.025em; }
.hero p { color:var(--muted); margin:0; font-size:1rem; line-height:1.75; max-width:76ch; }
.kpi { background:var(--surface); border-top:1px solid var(--line); border-radius:4px; padding:16px 18px; height:100%; }
.kpi .label { color:var(--muted); font-size:.85rem; }
.kpi .value { color:var(--ink); font-size:1.7rem; font-weight:650; margin-top:6px; font-variant-numeric:tabular-nums; }
.kpi .hint { color:var(--muted); font-size:.8rem; margin-top:6px; line-height:1.6; }
.mcard { background:var(--surface); border:1px solid var(--line); border-radius:6px; overflow:hidden;
    height:100%; display:flex; flex-direction:column;
    transition:border-color var(--t) var(--ease), transform var(--t) var(--ease); }
.mcard:hover { border-color:rgba(239,184,91,.5); transform:translateY(-2px); }
.mcover { position:relative; min-height:146px; display:flex; align-items:center; justify-content:center;
    border-bottom:2px solid var(--accent); }
.mcover .title { font-family:Georgia,"Microsoft YaHei",serif; font-size:1.1rem; font-weight:600; color:#fff;
    padding:40px 16px 16px; text-align:left; width:100%; line-height:1.38; overflow-wrap:anywhere;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
    text-shadow:0 1px 2px rgba(0,0,0,.25); }
.mcover .rank { position:absolute; top:9px; left:11px; font-size:.74rem; color:rgba(255,255,255,.82);
    letter-spacing:.06em; font-variant-numeric:tabular-nums; }
.mcover .score { position:absolute; top:8px; right:8px; font-size:.76rem; font-weight:700; color:#102820;
    background:var(--good); padding:3px 9px; border-radius:999px; font-variant-numeric:tabular-nums; }
.mcover .score.warn { background:var(--accent); color:#30230b; }
.mbody { padding:12px 14px 14px; display:flex; flex-direction:column; gap:6px; }
.mbody .mtitle { font-size:.94rem; font-weight:600; color:var(--ink); line-height:1.45; overflow-wrap:anywhere;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height:2.9em; }
.mbody .myear, .mbody .mgenres, .mbody .mmeta { color:var(--muted); font-size:.78rem; line-height:1.5; }
.mbody .mgenres { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mbody .mmeta { margin-top:2px; }
.mstats { display:flex; gap:10px; margin-top:auto; padding-top:9px; border-top:1px solid var(--line-soft); }
.mstats > div { flex:1; min-width:0; }
.mstats .k { display:block; color:var(--muted); font-size:.66rem; letter-spacing:.05em; }
.mstats .v { display:block; color:var(--ink); font-size:.92rem; font-weight:650; line-height:1.35;
    font-variant-numeric:tabular-nums; white-space:nowrap; }
.mstats .v.amber { color:var(--accent); }
/* 紧凑卡：用于侧栏式列表，不显示大海报 */
.mcard.compact { flex-direction:row; align-items:stretch; }
.mcard.compact .mc-bar { width:5px; flex:none; }
.mcard.compact .mc-body { padding:9px 11px; display:flex; flex-direction:column; gap:2px;
    min-width:0; justify-content:center; }
.mcard.compact .mc-title { font-size:.86rem; font-weight:600; color:var(--ink); line-height:1.35;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.mcard.compact .mc-meta { color:var(--muted); font-size:.7rem; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis; }
.mcard.compact .mc-rate { color:var(--accent); font-size:.74rem; font-weight:600;
    font-variant-numeric:tabular-nums; }
.mcard.compact .mc-head { display:flex; align-items:baseline; gap:6px; min-width:0; }
.mcard.compact .mc-rank { color:var(--muted); font-size:.7rem; letter-spacing:.04em; flex:none;
    font-variant-numeric:tabular-nums; }
.mcard.compact .mc-side { margin-left:auto; display:flex; flex-direction:column; justify-content:center;
    gap:2px; padding:9px 12px 9px 6px; flex:none; text-align:right; }
.mcard.compact .mc-side .k { color:var(--muted); font-size:.62rem; letter-spacing:.04em; }
.mcard.compact .mc-side .v { color:var(--ink); font-size:.82rem; font-weight:650;
    font-variant-numeric:tabular-nums; line-height:1.2; }
.mcard.compact .mc-side .v.amber { color:var(--accent); }
.mcard.compact .mc-side > div + div { margin-top:3px; }

.badge-fb { display:inline-block; margin-top:6px; font-size:.72rem; color:#30230b; background:var(--accent);
    padding:2px 8px; border-radius:999px; width:fit-content; }
table.matrix { border-collapse:separate; border-spacing:3px; font-size:.95rem; width:100%; }
table.matrix th { color:var(--muted); font-weight:600; padding:8px 4px; text-align:center; font-size:.85rem; }
table.matrix th.hl { color:var(--compare); }
table.matrix td { background:#23313e; color:var(--ink); text-align:center; padding:12px 4px; border-radius:3px; font-variant-numeric:tabular-nums; }
table.matrix td.missing { color:#96a5b0; background:#1a242e; }
table.matrix td.hl-row { background:#55452d; color:#fff; font-weight:600; }
table.matrix td.hl-col { background:#28514f; }
table.matrix td.hl-cell { background:var(--accent); color:#211a0e; font-weight:800; font-size:1.05rem; }
table.matrix td.hl-cell.is-hidden { outline:2px dashed #211a0e; outline-offset:-4px; }
table.matrix th.hl-name { color:var(--accent); font-weight:700; }
.stepbar { display:flex; gap:8px; margin:10px 0 20px; }
.stepbar .dot { flex:1; height:4px; border-radius:2px; background:var(--line); }
.stepbar .dot.on { background:var(--accent); }
.note,.warnbox { border-left:3px solid var(--compare); background:#1b3036; padding:14px 16px; border-radius:0 4px 4px 0; font-size:.9rem; color:var(--ink); line-height:1.7; }
.warnbox { border-color:var(--accent); background:#342d22; }
.section { font-size:1.15rem; font-weight:650; color:var(--ink); margin:28px 0 12px; }
.section .sub { display:inline-block; font-weight:400; color:var(--muted); font-size:.85rem; margin-left:12px; }
.stTabs [data-baseweb="tab"] { font-size:.95rem; min-height:44px; }
div[data-testid="stButton"] button { min-height:44px; border-radius:6px;
    transition:background-color var(--t) var(--ease), border-color var(--t) var(--ease),
               color var(--t) var(--ease), box-shadow var(--t) var(--ease); }
.stButton button:hover { border-color:var(--accent); color:var(--accent); }
section[data-testid="stSidebar"] .stRadio label { padding:9px 8px; border-radius:4px; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background:#293136; border-left:3px solid var(--accent); }
.brand { border-bottom:1px solid var(--line); padding:4px 0 24px; margin-bottom:18px; }
.brand-title { font-size:1.2rem; font-weight:700; color:var(--ink); }
.brand-subtitle { color:var(--muted); font-size:.85rem; margin-top:8px; }

/* ---------- 课堂演示（与完整实验共用同一套 token 与控件） ---------- */
.dm-head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.dm-brand { font-size:1.8rem; font-weight:700; color:var(--ink); letter-spacing:-.015em; }
.dm-mode { color:var(--muted); font-size:.85rem; margin-top:4px; }
.dm-link { margin-top:6px; }
.dm-link a { color:var(--muted); font-size:.78rem; text-decoration:none;
    border-bottom:1px dotted var(--line); }
.dm-link a:hover { color:var(--accent); border-bottom-color:var(--accent); }
.dm-pill { border:1px solid var(--line); color:var(--muted); border-radius:999px;
    padding:5px 12px; font-size:.78rem; white-space:nowrap; }
.dm-tabs { display:flex; gap:10px; margin:10px 0 12px; padding-bottom:8px;
    border-bottom:1px solid var(--line); }
.dm-tab { flex:1; text-align:center; padding:9px 8px; border-radius:5px;
    color:var(--muted); font-size:.9rem; }
.dm-tab.on { background:var(--surface); color:var(--accent); font-weight:650;
    box-shadow:inset 3px 0 0 var(--accent); }
.dm-q { font-size:1.32rem; font-weight:700; color:var(--ink); margin:0 0 2px; letter-spacing:-.01em; }
.dm-sub { color:var(--muted); font-size:.9rem; margin-bottom:10px; }
.dm-cards { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.dm-cols { display:grid; grid-template-columns:1fr 1fr; gap:28px; }
.dm-col-name { font-size:1.04rem; font-weight:700; color:var(--ink); }
.dm-col-name.u { color:var(--accent); }
.dm-col-name.i { color:var(--compare); }
.dm-col-desc { color:var(--muted); font-size:.82rem; margin:3px 0 6px; }
.dm-row { display:flex; align-items:center; gap:10px; padding:10px 4px;
    border-bottom:1px solid var(--line); }
.dm-row:last-child { border-bottom:none; }
.dm-rank { color:var(--muted); font-size:.78rem; width:18px; font-variant-numeric:tabular-nums; }
.dm-name { flex:1; font-size:.97rem; color:var(--ink); overflow-wrap:anywhere; }
.dm-meta { color:var(--muted); font-size:.78rem; font-variant-numeric:tabular-nums; }
.dm-chip { font-size:.72rem; padding:2px 9px; border-radius:999px; white-space:nowrap;
    font-variant-numeric:tabular-nums; }
.dm-chip.same { background:rgba(136,211,178,.16); color:#88d3b2; }
.dm-chip.u { background:rgba(239,184,91,.16); color:var(--accent); }
.dm-chip.i { background:rgba(103,198,192,.16); color:var(--compare); }
.dm-chip.pos { background:rgba(136,211,178,.16); color:#88d3b2; }
.dm-chip.neg { background:rgba(224,122,122,.18); color:#e07a7a; }
.dm-film { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
    border-radius:6px; padding:11px 14px; display:flex; gap:12px; align-items:center; margin-bottom:8px; }
.dm-film .t { font-family:Georgia,"Microsoft YaHei",serif; font-size:1.05rem;
    font-weight:600; color:var(--ink); }
.dm-film .g { color:var(--muted); font-size:.8rem; margin-top:3px; }
.dm-concl { margin-top:8px; padding:8px 14px; background:var(--surface);
    border-left:3px solid var(--accent); border-radius:0 5px 5px 0;
    color:var(--ink); font-size:.92rem; line-height:1.6; }
.dm-hr { border:none; border-top:1px solid var(--line); margin:8px 0 10px; }

/* ---------- 证据头部（推荐页单条解释） ---------- */
.evhead { background:var(--surface); border:1px solid var(--line); border-radius:6px;
    padding:12px 14px; margin-bottom:8px; }
.evhead .evtitle { font-family:Georgia,"Microsoft YaHei",serif; font-size:1.05rem;
    font-weight:600; color:var(--ink); line-height:1.4; }
.evhead .evyear { color:var(--muted); font-family:"Segoe UI","Microsoft YaHei",sans-serif;
    font-size:.82rem; font-weight:400; margin-left:2px; }
.evhead .evgenre { color:var(--muted); font-size:.78rem; margin-top:3px; }
.evhead .evmetrics { display:flex; gap:18px; margin-top:10px; padding-top:9px;
    border-top:1px solid var(--line-soft); }
.evhead .evmetrics > div { min-width:0; }
.evhead .k { display:block; color:var(--muted); font-size:.66rem; letter-spacing:.05em; }
.evhead .v { display:block; color:var(--ink); font-size:.98rem; font-weight:650;
    font-variant-numeric:tabular-nums; line-height:1.35; }
.evhead .v.amber { color:var(--accent); }
.evhead .evfb { margin-top:9px; color:var(--accent); font-size:.76rem; }

.evlist { border:1px solid var(--line); border-radius:6px; padding:2px 12px; background:var(--surface); }
.evlist .dm-row:last-child { border-bottom:none; }

/* ---------- 交互状态与精修 ---------- */
.stTabs [data-baseweb="tab"] { transition:color var(--t) var(--ease); }
.stTabs [data-baseweb="tab"]:hover { color:var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:6px; overflow:hidden;
    transition:border-color var(--t) var(--ease); }
[data-testid="stExpander"]:hover { border-color:#42576b; }
[data-testid="stExpander"] summary { transition:color var(--t) var(--ease); }
[data-testid="stExpander"] summary:hover { color:var(--ink); }
.modebar { display:none !important; }
.dm-tab { transition:background-color var(--t) var(--ease), color var(--t) var(--ease), box-shadow var(--t) var(--ease); }
.dm-row { transition:background-color 150ms var(--ease); border-radius:4px; }
.dm-row:hover { background:rgba(255,255,255,.028); }
.dm-row.same { box-shadow:inset 2px 0 0 rgba(136,211,178,.55); }
.dm-bar { width:52px; height:4px; border-radius:2px; background:rgba(255,255,255,.09);
    overflow:hidden; flex:none; }
.dm-bar i { display:block; height:100%; background:var(--good); }
.dm-bar i.neg { background:var(--bad); }
.dm-chip { transition:background-color 150ms var(--ease); }
a:focus-visible, button:focus-visible, input:focus-visible, summary:focus-visible,
[role="tab"]:focus-visible, [role="radio"]:focus-visible {
    outline:2px solid var(--accent); outline-offset:3px; }

/* ---------- 按钮：两种模式共用同一套外观 ---------- */
div[data-testid="stButton"] button[kind="primary"] { background:var(--accent);
    border:1px solid var(--accent); color:#211a0e; font-weight:600; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#f3c877;
    border-color:#f3c877; color:#211a0e; }
div[data-testid="stButton"] button[kind="secondary"] { background:transparent;
    border:1px solid var(--line); color:var(--ink); }
div[data-testid="stButton"] button[kind="secondary"]:hover { border-color:var(--accent);
    color:var(--accent); }
div[data-testid="stButton"] button:disabled { opacity:.42; }
@media(max-width:640px) {
 .block-container { padding:3rem 1rem 2rem; }
 .hero { padding-left:16px; }
 .section .sub { display:block; margin:4px 0 0; }
 .kpi { padding:12px 14px; }
 table.matrix th { font-size:.72rem; overflow-wrap:anywhere; }
 .dm-cards { grid-template-columns:1fr; }
 .dm-cols { grid-template-columns:1fr; gap:18px; }
}
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } }
</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, kicker: str, desc: str) -> None:
    import streamlit as st
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div>'
        f"<h1>{title}</h1><p>{desc}</p></div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items = [(label, value, hint), ...]"""
    import streamlit as st
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        col.markdown(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="hint">{hint}</div></div>',
            unsafe_allow_html=True,
        )


def section(title: str, sub: str = "") -> None:
    import streamlit as st
    st.markdown(f'<div class="section">{title}<span class="sub">{sub}</span></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    import streamlit as st
    cls = "warnbox" if warn else "note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _gradient(key: int) -> tuple[str, str]:
    return GRADIENTS[abs(hash(str(key))) % len(GRADIENTS)]


def movie_card(
    title: str, year, genres: str, score: float, meta: str = "",
    rank: int | None = None, fallback: bool = False, item_id: int = 0, cover_text: str | None = None,
    score_text: str | None = None, stats: list[tuple[str, str]] | None = None,
    compact: bool = False,
) -> str:
    """电影卡片。

    ``stats`` 用于把“推荐分 / 预测分 / 邻居数”拆成带标签的小格，避免混在一行小字里；
    不传时退化为单行 ``meta``，与旧调用保持兼容。
    """
    c1, c2 = _gradient(item_id)
    if compact:                                   # 列表场景：色条 + 标题 + 元信息（+ 右侧指标）
        year_txt = f"{int(year)} · " if pd.notna(year) else ""
        rank_html = f'<span class="mc-rank">#{rank:02d}</span>' if rank else ""
        if stats:
            side = "".join(
                f'<div><span class="k">{k}</span><span class="v{" amber" if i == len(stats) - 1 else ""}">'
                f'{v}</span></div>'
                for i, (k, v) in enumerate(stats)
            )
            tail = f'<div class="mc-side">{side}</div>'
            rate = ""
        else:
            tail = ""
            rate = f'<div class="mc-rate">★ {score:.0f}</div>'
        return (
            f'<div class="mcard compact"><div class="mc-bar" style="background:{c1}"></div>'
            f'<div class="mc-body"><div class="mc-head">{rank_html}'
            f'<span class="mc-title">{title}</span></div>'
            f'<div class="mc-meta">{year_txt}{genres}</div>{rate}</div>{tail}</div>'
        )
    badge_cls = "score warn" if fallback else "score"
    score_txt = score_text or ("均值回退" if fallback else f"{score:.2f}")
    cover = cover_text or (title[:12] + ("…" if len(title) > 12 else ""))
    year_txt = f'<span class="myear">({int(year)})</span>' if pd.notna(year) else ""
    rank_txt = f'<div class="rank">#{rank:02d}</div>' if rank else ""
    fb = '<span class="badge-fb">用户均值回退</span>' if fallback else ""
    if stats:
        cells = "".join(
            f'<div><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in stats
        )
        tail = f'<div class="mstats">{cells}</div>{fb}'
    else:
        tail = (f'<div class="mmeta">{meta}</div>' if meta else "") + fb
    return (
        f'<div class="mcard"><div class="mcover" style="background:{c1}">'
        f'{rank_txt}<div class="{badge_cls}">{score_txt}</div>'
        f'<div class="title">{cover}</div></div>'
        f'<div class="mbody"><div class="mtitle">{title} {year_txt}</div>'
        f'<div class="mgenres">{genres}</div>{tail}</div></div>'
    )


def matrix_table(df: pd.DataFrame, user: int, item: int, hidden: bool = False) -> str:
    """评分矩阵：高亮目标用户所在行、目标电影所在列，目标格子显示 ?。

    ``hidden=True`` 表示该格原本有评分、但为了演示被临时隐藏（用虚线圈标出）。
    """
    cols = list(df.columns)
    head = "".join(
        f'<th class="{"hl" if j == item else ""}">{c}</th>' for j, c in enumerate(cols)
    )
    body = []
    for u, name in enumerate(df.index):
        cells = []
        for j in range(len(cols)):
            v = df.iloc[u, j]
            cls = []
            if u == user:
                cls.append("hl-row")
            if j == item:
                cls.append("hl-col")
            if u == user and j == item:
                cls.append("hl-cell")
                if hidden:
                    cls.append("is-hidden")
                txt = "?"
            elif pd.isna(v):
                cls.append("missing")
                txt = "—"
            else:
                txt = f"{v:.0f}"
            cells.append(f'<td class="{" ".join(cls)}">{txt}</td>')
        body.append(f'<tr><th class="{"hl-name" if u == user else ""}">{name}</th>{"".join(cells)}</tr>')
    return (
        f'<table class="matrix"><thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def step_progress(current: int, total: int) -> str:
    dots = "".join(f'<div class="dot {"on" if i <= current else ""}"></div>' for i in range(total))
    return f'<div class="stepbar">{dots}</div>'


def genre_gradient_style(genres: str) -> str:
    h = int(hashlib.md5(genres.encode("utf-8")).hexdigest()[:6], 16)
    c1, c2 = GRADIENTS[h % len(GRADIENTS)]
    return f"linear-gradient(135deg,{c1},{c2})"
