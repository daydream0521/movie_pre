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
 --canvas:#141c24; --surface:#1c2833; --line:#344451;
 --ink:#edf1f2; --muted:#a7b6c1; --accent:#efb85b; --compare:#67c6c0;
}
html, body, .stApp, input, button, textarea, select {
 font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
}
.stApp { background:var(--canvas); color:var(--ink); }
section[data-testid="stSidebar"] { background:#101820; border-right:1px solid var(--line); }
#MainMenu, footer, header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
.block-container { padding-top:2.8rem; padding-bottom:3rem; max-width:1280px; }
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
.mcard { background:var(--surface); border:1px solid var(--line); border-radius:5px; overflow:hidden; height:100%; }
.mcover { position:relative; min-height:150px; display:flex; align-items:center; justify-content:center; border-bottom:3px solid var(--accent); }
.mcover .title { font-family:Georgia,"Microsoft YaHei",serif; font-size:1.2rem; font-weight:600; color:#fff; padding:42px 16px 18px; text-align:left; width:100%; line-height:1.4; overflow-wrap:anywhere; }
.mcover .rank { position:absolute; top:10px; left:12px; font-size:.8rem; color:#fff; }
.mcover .score { position:absolute; top:8px; right:8px; font-size:.8rem; font-weight:700; color:#102820; background:#88d3b2; padding:3px 8px; border-radius:3px; }
.mcover .score.warn { background:#efb85b; color:#30230b; }
.mbody { padding:14px; }
.mbody .mtitle { font-size:.95rem; font-weight:600; color:var(--ink); line-height:1.5; overflow-wrap:anywhere; }
.mbody .myear, .mbody .mgenres, .mbody .mmeta { color:var(--muted); font-size:.8rem; line-height:1.6; }
.mbody .mgenres { margin-top:6px; min-height:2.1em; }
.mbody .mmeta { margin-top:8px; }
.badge-fb { display:inline-block; margin-top:8px; font-size:.75rem; color:#30230b; background:#efb85b; padding:2px 7px; border-radius:3px; }
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
.stButton button { min-height:44px; border-radius:5px; transition:background .15s; }
.stButton button:hover { border-color:var(--accent); color:var(--accent); }
button:focus-visible, input:focus-visible, a:focus-visible { outline:2px solid var(--accent); outline-offset:3px; }
section[data-testid="stSidebar"] .stRadio label { padding:9px 8px; border-radius:4px; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background:#293136; border-left:3px solid var(--accent); }
.brand { border-bottom:1px solid var(--line); padding:4px 0 24px; margin-bottom:18px; }
.brand-title { font-size:1.2rem; font-weight:700; color:var(--ink); }
.brand-subtitle { color:var(--muted); font-size:.85rem; margin-top:8px; }
@media(max-width:640px) {
 .block-container { padding:3rem 1rem 2rem; }
 .hero { padding-left:16px; }
 .section .sub { display:block; margin:4px 0 0; }
 .kpi { padding:12px 14px; }
 table.matrix th { font-size:.72rem; overflow-wrap:anywhere; }
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
    score_text: str | None = None,
) -> str:
    c1, c2 = _gradient(item_id)
    badge_cls = "score warn" if fallback else "score"
    score_txt = score_text or ("均值回退" if fallback else f"{score:.2f}")
    cover = cover_text or (title[:12] + ("…" if len(title) > 12 else ""))
    year_txt = f'<span class="myear">({int(year)})</span>' if pd.notna(year) else ""
    rank_txt = f'<div class="rank">#{rank:02d}</div>' if rank else ""
    meta_html = f'<div class="mmeta">{meta}</div>' if meta else ""
    fb = '<span class="badge-fb">用户均值回退</span>' if fallback else ""
    return (
        f'<div class="mcard"><div class="mcover" style="background:{c1}">'
        f'{rank_txt}<div class="{badge_cls}">{score_txt}</div>'
        f'<div class="title">{cover}</div></div>'
        f'<div class="mbody"><div class="mtitle">{title} {year_txt}</div>'
        f'<div class="mgenres">{genres}</div>{meta_html}{fb}</div></div>'
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
