"""用 Streamlit 官方 AppTest 无头跑通课堂演示与完整实验，确认没有异常、关键元素存在。

用法：python tests/test_app_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def _run() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def run_page(page: str) -> AppTest:
    """切到「完整实验」并打开指定页面。"""
    at = _run()
    at.sidebar.radio(key="nav_mode").set_value("完整实验").run()
    at.sidebar.radio(key="full_page").set_value(page).run()
    assert not at.exception, f"{page} 页面抛出异常：{[e.value for e in at.exception]}"
    return at


def main() -> int:
    # ---------------------------------------------------------- 课堂演示（默认）
    at = _run()
    assert at.radio(key="nav_mode").value == "课堂演示", "默认没有进入课堂演示"
    assert at.button(key="demo_prev").disabled is True, "第 1 步的上一步应当禁用"
    assert at.button(key="demo_next").label.startswith("下一步")
    text = "".join(m.value for m in at.markdown)
    assert "RMSE" not in text and "覆盖率" not in text, "默认页不应出现指标类信息"
    print("[OK] 课堂演示：默认进入第 1 步，主按钮只有上一步/下一步，未暴露指标")

    at.button(key="demo_next").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.button(key="demo_prev").disabled is False
    print("[OK] 课堂演示：第 2 步（两种算法并排）无异常")

    at.button(key="demo_next").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.button(key="demo_restart").label == "重新演示"
    text3 = "".join(m.value for m in at.markdown)
    assert "查看计算过程" in text3 or "User-CF：人的证据" in text3
    print("[OK] 课堂演示：第 3 步（解释原因）无异常，按钮变为「重新演示」")

    at.button(key="demo_restart").click().run()
    assert at.button(key="demo_next").label.startswith("下一步")
    print("[OK] 课堂演示：重新演示回到第 1 步")

    # ---------------------------------------------------------- 完整实验
    for page in ["概览", "原理实验室", "电影推荐", "实验结果", "从 CF 到深度学习"]:
        at = run_page(page)
        print(f"[OK] {page:12s} markdown={len(at.markdown)} dataframe={len(at.dataframe)} "
              f"charts={len(at.get('plotly_chart')) if hasattr(at, 'get') else '-'}")

    # 交互：切换算法 + 改变 K + 推进一步
    at = run_page("原理实验室")
    at.radio(key="lab_algo").set_value("Item-CF").run()
    assert not at.exception, [e.value for e in at.exception]
    at.slider(key="lab_k").set_value(1).run()
    assert not at.exception
    at.button[1].click().run()          # 下一步
    assert not at.exception
    print("[OK] 原理实验室：切换 Item-CF / K=1 / 下一步 均无异常")

    # 目标电影已被打分：应走「隐藏该评分」的分支，而不是当成空白格
    at.selectbox(key="lab_user").set_value(0).run()
    at.selectbox(key="lab_item").set_value(0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("隐藏" in m.value for m in at.markdown), "已评分目标没有给出隐藏说明"
    print("[OK] 原理实验室：目标电影已评分 → 隐藏该评分并说明原因")

    at = run_page("电影推荐")
    at.radio(key="rec_algo").set_value("Item-CF").run()
    assert not at.exception
    at.select_slider(key="rec_k").set_value(5).run()
    assert not at.exception
    print("[OK] 电影推荐：切换算法 / K=5 均无异常")

    # 随机换用户：按钮必须真的改掉 selectbox 的值（曾因改错 session_state 键而失效）
    before = at.selectbox(key="rec_user_select").value
    btn = [b for b in at.button if "随机" in b.label][0]
    btn.click().run()
    after = at.selectbox(key="rec_user_select").value
    assert after != before, f"随机换用户没有生效（仍为 {before}）"
    print(f"[OK] 电影推荐：随机换用户 {before} → {after}")

    for view in ("双算法对比", "同一部电影，两种解释"):
        at.radio(key="rec_view").set_value(view).run()
        assert not at.exception, [e.value for e in at.exception]
    print("[OK] 电影推荐：对比视图 / 同片双解释 均无异常")

    print("全部页面烟测通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
