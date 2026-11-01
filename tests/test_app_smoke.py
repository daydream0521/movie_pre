"""用 Streamlit 官方 AppTest 无头跑通每个页面，确认没有异常、关键元素存在。

用法：python tests/test_app_smoke.py
（pytest 也可直接收集，见 tests/test_app.py）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def run_page(page: str, extra_widgets: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    if extra_widgets:
        for k, v in extra_widgets.items():
            at.session_state[k] = v
        at.run()
    assert not at.exception, f"{page} 页面抛出异常：{[e.value for e in at.exception]}"
    return at


def main() -> int:
    for page in ["概览", "原理实验室", "电影推荐", "实验结果", "从 CF 到深度学习"]:
        at = run_page(page)
        n_md = len(at.markdown)
        print(f"[OK] {page:12s} markdown={n_md} dataframe={len(at.dataframe)} "
              f"charts/other={len(at.get('plotly_chart')) if hasattr(at, 'get') else '-'}")

    # 交互：切换算法 + 改变 K + 推进一步
    at = run_page("原理实验室")
    at.radio(key="lab_algo").set_value("Item-CF").run()
    assert not at.exception, [e.value for e in at.exception]
    at.slider(key="lab_k").set_value(1).run()
    assert not at.exception
    at.button[1].click().run()          # 下一步
    assert not at.exception
    print("[OK] 原理实验室：切换 Item-CF / K=1 / 下一步 均无异常")

    at = run_page("电影推荐")
    at.radio(key="rec_algo").set_value("Item-CF").run()
    assert not at.exception
    at.select_slider(key="rec_k").set_value(5).run()
    assert not at.exception
    print("[OK] 电影推荐：切换算法 / K=5 均无异常")

    print("全部页面烟测通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
