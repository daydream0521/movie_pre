"""下载并解压 MovieLens 100K。

用法：
    python scripts/download_data.py            # 下载（支持断点续传）
    python scripts/download_data.py --check    # 只检查本地数据是否就绪

数据集许可：MovieLens 数据由 GroupLens 发布，仅用于研究/教学；
本仓库不把原始数据提交进 Git（见 .gitignore）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "ml-100k.zip"
TARGET = DATA_DIR / "ml-100k"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
REQUIRED = ["u.data", "u.item", "u1.base", "u1.test"]


def is_ready() -> bool:
    return all((TARGET / f).exists() for f in REQUIRED)


def download(url: str = URL, dst: Path = ZIP_PATH, chunk: int = 65536, retries: int = 6) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "movielens-explorer/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "ab") as fh:
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    have += len(buf)
                    if total:
                        pct = 100 * have / total
                        print(f"\r  下载中 {have/1e6:.2f}/{total/1e6:.2f} MB ({pct:5.1f}%)", end="")
            print()
            if total is None or dst.stat().st_size >= total:
                return
        except Exception as exc:                      # 网络中断：保留已下载部分，重试续传
            print(f"\n  第 {attempt} 次尝试中断：{exc}")
            if attempt == retries:
                raise
    return


def extract() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    if not is_ready():
        raise RuntimeError("解压后未找到 u.data / u.item 等文件，请检查压缩包是否完整。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查数据是否就绪")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    if is_ready():
        print(f"数据已就绪：{TARGET}")
        return 0
    if args.check:
        print(f"数据未就绪。请运行 python scripts/download_data.py（或手动下载到 {TARGET}）")
        return 1

    print(f"从 {args.url} 下载 MovieLens 100K ...")
    download(args.url)
    print("解压中 ...")
    extract()
    print(f"完成：{TARGET}")

    if "--force-clean" in sys.argv and ZIP_PATH.exists():
        ZIP_PATH.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
