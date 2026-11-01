"""用 Playwright 驱动本机 Chrome 给各页面截图（仅用于人工检查视觉效果）。"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".shots"
OUT.mkdir(exist_ok=True)
BASE = "http://localhost:8501"

PAGES = sys.argv[1:] or ["overview", "lab", "recommend", "results", "concept"]


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 2600}, device_scale_factor=1.2)
        for name in PAGES:
            url = f"{BASE}/?page={name}"
            page.goto(url, wait_until="load", timeout=90000)
            # 等 Streamlit 把骨架屏换成真实内容
            page.wait_for_selector('[data-testid="stAppViewContainer"] .block-container', timeout=90000)
            page.wait_for_function(
                "() => !document.querySelector('[data-testid=\"stSkeleton\"]') "
                "&& document.querySelectorAll('.stMarkdown, [data-testid=\"stVerticalBlock\"]').length > 2",
                timeout=90000,
            )
            page.wait_for_timeout(3500)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print("saved", name)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
