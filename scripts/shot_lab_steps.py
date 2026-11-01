"""按步骤点击原理实验室并截图，验证每一步的表格/图表都能渲染。"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".shots"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.4)
        page.goto("http://localhost:8501/?page=lab", wait_until="load", timeout=90000)
        page.wait_for_selector(".matrix", timeout=90000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / "lab_step1.png"), full_page=True)

        for i in (2, 3, 4):
            page.get_by_role("button", name="下一步 →").click()
            page.wait_for_timeout(2200)
            page.screenshot(path=str(OUT / f"lab_step{i}.png"), full_page=True)
            print("step", i, "ok")

        # 切到 Item-CF 再看一次第 4 步
        page.get_by_text("Item-CF", exact=True).first.click()
        page.wait_for_timeout(2500)
        for _ in range(3):
            try:
                page.get_by_role("button", name="下一步 →").click()
                page.wait_for_timeout(1200)
            except Exception:
                break
        page.screenshot(path=str(OUT / "lab_item_step4.png"), full_page=True)
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
