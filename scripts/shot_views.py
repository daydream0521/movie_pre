"""生成 docs/screenshots/ 下的正式截图（页面 + 关键交互状态）。"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"


def shot(pg, name: str) -> None:
    pg.screenshot(path=str(OUT / name), full_page=True)
    print("saved", name)


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1500, "height": 1900}, device_scale_factor=1.2)

        # 概览
        pg.goto(f"{BASE}/?page=overview", wait_until="load", timeout=90000)
        pg.wait_for_selector("text=评分分布", timeout=90000)
        pg.wait_for_timeout(3000)
        shot(pg, "01-overview.png")

        # 实验室：默认（空白目标）与逐步
        pg.goto(f"{BASE}/?page=lab", wait_until="load", timeout=90000)
        pg.wait_for_selector(".matrix", timeout=90000)
        pg.wait_for_timeout(2000)
        shot(pg, "02-lab-step1-matrix.png")
        for i in (2, 3, 4):
            pg.get_by_role("button", name="下一步 →").click()
            pg.wait_for_timeout(1800)
            if i == 4:
                shot(pg, "03-lab-step4-weights.png")

        # 实验室：目标已评分 → 隐藏分支
        try:
            pg.goto(f"{BASE}/?page=lab", wait_until="load", timeout=90000)
            pg.wait_for_selector(".matrix", timeout=90000)
            pg.wait_for_timeout(1500)
            pg.get_by_role("combobox").nth(1).click()
            pg.wait_for_timeout(600)
            pg.get_by_role("option", name="T1《星际穿越》").click()
            pg.wait_for_timeout(2500)
            shot(pg, "04-lab-hidden-rating.png")
        except Exception as exc:                      # 选择器变动时不要让整个流程失败
            print("隐藏分支截图跳过：", exc)

        # 推荐页：单算法
        pg.goto(f"{BASE}/?page=recommend", wait_until="load", timeout=90000)
        pg.wait_for_selector("text=Top-10 推荐", timeout=90000)
        pg.wait_for_timeout(2500)
        shot(pg, "05-recommend-single.png")

        # 推荐页：双算法对比
        pg.get_by_text("双算法对比", exact=True).click()
        pg.wait_for_timeout(3000)
        shot(pg, "06-recommend-compare.png")

        # 推荐页：同一部电影两种解释
        pg.get_by_text("同一部电影，两种解释", exact=True).click()
        pg.wait_for_timeout(3000)
        shot(pg, "07-recommend-same-movie.png")

        # 实验结果页
        pg.goto(f"{BASE}/?page=results", wait_until="load", timeout=90000)
        pg.wait_for_selector('[data-testid="stTabs"]', timeout=90000)
        pg.wait_for_timeout(2500)
        shot(pg, "08-results-e1.png")
        pg.get_by_role("tab", name="E4 Top-10 推荐质量").click()
        pg.wait_for_timeout(2500)
        shot(pg, "09-results-e4.png")

        # 概念页
        pg.goto(f"{BASE}/?page=concept", wait_until="load", timeout=90000)
        pg.wait_for_timeout(3000)
        shot(pg, "10-concept.png")

        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
