"""截取课堂演示三步在两种投屏尺寸下的首屏，并记录是否需要滚动。"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [(1366, 768), (1920, 1080)]


def measure(pg) -> dict:
    return pg.evaluate(
        """() => {
            const el = document.querySelector('section.main')
                    || document.querySelector('[data-testid="stMain"]')
                    || document.querySelector('.stApp');
            const doc = document.scrollingElement;
            return {
                el: el ? el.scrollHeight : null,
                elVisible: el ? el.clientHeight : null,
                doc: doc ? doc.scrollHeight : null,
                win: window.innerHeight
            };
        }"""
    )


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        for w, h in SIZES:
            pg = b.new_page(viewport={"width": w, "height": h})
            pg.goto("http://localhost:8501/", wait_until="load", timeout=90000)
            pg.wait_for_selector(".dm-q", timeout=90000)
            pg.wait_for_timeout(2200)
            for step in (1, 2, 3):
                name = f"demo-{w}x{h}-step{step}.png"
                pg.screenshot(path=str(OUT / name))
                m = measure(pg)
                scroll = (m["el"] or 0) - (m["elVisible"] or 0)
                print(f"{name:34s} 内容高 {m['el']} / 可视 {m['elVisible']} → "
                      f"{'需滚动 ' + str(scroll) + 'px' if scroll > 2 else '无需滚动'}")
                if step < 3:
                    pg.get_by_role("button", name="下一步 →").click()
                    pg.wait_for_timeout(1600)
            pg.close()
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
