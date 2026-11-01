"""界面层：配色、卡片、图表等可复用渲染片段（全部离线，不依赖海报网络图片）。"""

from __future__ import annotations

import hashlib

import pandas as pd

ACCENT = "#3ba7ff"
ACCENT_2 = "#7c5cff"
GOOD = "#34d399"
WARN = "#fbbf24"

GRADIENTS = [
    ("#1e3a8a", "#0ea5e9"),
    ("#4c1d95", "#7c3aed"),
    ("#0f766e", "#14b8a6"),
    ("#7c2d12", "#f59e0b"),
    ("#831843", "#ec4899"),
    ("#1e293b", "#38bdf8"),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
}
.stApp {
    background: radial-gradient(1100px 520px at 12% -8%, #16263f 0%, #0a1122 58%) fixed;
    color: #e9eefb;
}
section[data-testid="stSidebar"] {
    background: #0c1526;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
#MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

h1, h2, h3, h4 { color: #f2f6ff !important; letter-spacing: -0.01em; }
p, li, span, div { color: #dbe4f5; }

/* ---------- 顶部横幅 ---------- */
.hero {
    background: linear-gradient(120deg, rgba(59,167,255,0.16), rgba(124,92,255,0.10) 45%, rgba(10,17,34,0) 78%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.hero .kicker {
    color: #7fd0ff; font-size: 0.78rem; letter-spacing: 0.18em;
    text-transform: uppercase; font-weight: 600; margin-bottom: 6px;
}
.hero h1 { font-size: 1.72rem; margin: 0 0 6px 0; font-weight: 700; }
.hero p { color: #9fb0cf; margin: 0; font-size: 0.95rem; line-height: 1.55; }

/* ---------- KPI ---------- */
.kpi {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 14px 16px; height: 100%;
}
.kpi .label { color: #93a4c4; font-size: 0.78rem; letter-spacing: 0.04em; }
.kpi .value { color: #f4f8ff; font-size: 1.42rem; font-weight: 700; margin-top: 4px; }
.kpi .hint { color: #7c8dab; font-size: 0.74rem; margin-top: 2px; }

/* ---------- 电影卡片 ---------- */
.mcard {
    background: #131f36; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; overflow: hidden; height: 100%;
    transition: transform .15s ease, border-color .15s ease;
}
.mcard:hover { transform: translateY(-3px); border-color: rgba(59,167,255,0.45); }
.mcover { position: relative; height: 118px; display: flex; align-items: center; justify-content: center; }
.mcover .title { font-size: 0.92rem; font-weight: 700; color: #ffffff; opacity: .92;
    padding: 0 10px; text-align: center; line-height: 1.25; text-shadow: 0 1px 8px rgba(0,0,0,.5); }
.mcover .rank { position: absolute; top: 8px; left: 10px; font-size: 0.72rem; font-weight: 700;
    color: rgba(255,255,255,.85); background: rgba(0,0,0,.32); padding: 1px 7px; border-radius: 8px; }
.mcover .score { position: absolute; top: 8px; right: 8px; font-size: 0.74rem; font-weight: 700;
    color: #06231a; background: #34d399; padding: 2px 8px; border-radius: 9px; }
.mcover .score.warn { background: #fbbf24; color: #2b1d00; }
.mbody { padding: 10px 12px 12px; }
.mbody .mtitle { font-size: 0.86rem; font-weight: 600; color: #eef4ff; line-height: 1.3; }
.mbody .myear { color: #7f91b0; font-weight: 400; font-size: 0.78rem; }
.mbody .mgenres { color: #8aa0c2; font-size: 0.73rem; margin-top: 3px; line-height: 1.35;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.1em; }
.mbody .mmeta { color: #6f82a3; font-size: 0.71rem; margin-top: 6px; }
.badge-fb { display:inline-block; margin-top:6px; font-size:0.68rem; color:#2b1d00; background:#fbbf24;
    padding:1px 7px; border-radius:8px; font-weight:600; }

/* ---------- 教学矩阵 ---------- */
table.matrix { border-collapse: separate; border-spacing: 3px; font-size: 0.84rem; width: 100%; }
table.matrix th { color: #93a4c4; font-weight: 600; padding: 6px 8px; text-align: center; font-size: 0.76rem; }
table.matrix th.hl { color: #c9b8ff; }
table.matrix td { background: #14203a; color: #cfdcf2; text-align: center; padding: 9px 6px;
    border-radius: 8px; font-variant-numeric: tabular-nums; min-width: 54px; }
table.matrix td.missing { color: #4d5f7d; background: #101a2e; }
table.matrix td.hl-row { background: rgba(59,167,255,0.30); color: #ffffff; font-weight: 600; }
table.matrix td.hl-col { background: rgba(124,92,255,0.26); }
table.matrix td.hl-cell { background: #3ba7ff; color: #04141f; font-weight: 800; font-size: 0.95rem; }
table.matrix th.hl-name { color: #7fd0ff; font-weight: 700; }

/* ---------- 步骤条 ---------- */
.stepbar { display: flex; gap: 8px; margin: 6px 0 16px; flex-wrap: wrap; }
.stepbar .dot { width: 100%; height: 4px; border-radius: 4px; background: rgba(255,255,255,0.10); }
.stepbar .dot.on { background: linear-gradient(90deg, #3ba7ff, #7c5cff); }

/* ---------- 说明块 ---------- */
.note {
    border-left: 3px solid #3ba7ff; background: rgba(59,167,255,0.08);
    padding: 10px 14px; border-radius: 0 10px 10px 0; font-size: 0.86rem; color: #c8d6ee;
}
.warnbox {
    border-left: 3px solid #fbbf24; background: rgba(251,191,36,0.08);
    padding: 10px 14px; border-radius: 0 10px 10px 0; font-size: 0.86rem; color: #e8d9b0;
}
.section { font-size: 1.06rem; font-weight: 700; color: #eaf1ff; margin: 20px 0 8px; }
.section .sub { font-weight: 400; color: #7f91b0; font-size: 0.8rem; margin-left: 8px; }
div[data-testid="stMetricValue"] { font-size: 1.3rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
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
        f'<div class="mcard"><div class="mcover" style="background:linear-gradient(135deg,{c1},{c2})">'
        f'{rank_txt}<div class="{badge_cls}">{score_txt}</div>'
        f'<div class="title">{cover}</div></div>'
        f'<div class="mbody"><div class="mtitle">{title} {year_txt}</div>'
        f'<div class="mgenres">{genres}</div>{meta_html}{fb}</div></div>'
    )


def matrix_table(df: pd.DataFrame, user: int, item: int) -> str:
    """评分矩阵：高亮目标用户所在行、目标电影所在列，目标格子显示 ?。"""
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
