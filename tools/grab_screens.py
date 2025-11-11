# tools/grab_screens.py
import asyncio, csv, hashlib, os, sys, time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from playwright.async_api import async_playwright
from tqdm import tqdm

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/urls_clean.csv"
OUT_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("images")
VIEWPORT = {"width": 1366, "height": 768}
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 6
TIMEOUT_MS = 15000   # 15s per page
WAIT_MS = 1500       # extra settle time after load

def safe_name(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    host = urlparse(url).hostname or "nohost"
    return f"{host}-{h}.png"

async def worker(play, rows):
    browser = await play.chromium.launch(headless=True, args=["--disable-gpu","--no-sandbox"])
    ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
    page = await ctx.new_page()
    # Reduce noise
    await ctx.route("**/*", lambda route: route.continue_())
    for url, label in rows:
        out_dir = OUT_ROOT / ("phish" if label==1 else "legit")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / safe_name(url)
        if out_fp.exists(): 
            continue
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            # Try to wait for network to be quiet, but don’t block forever
            try:
                await page.wait_for_load_state("networkidle", timeout=4000)
            except:
                pass
            await page.wait_for_timeout(WAIT_MS)
            await page.screenshot(path=str(out_fp), full_page=False)  # viewport shot
        except Exception as e:
            # record failures
            (OUT_ROOT / "failed.txt").open("a", encoding="utf-8").write(f"{label}\t{url}\t{e}\n")
    await ctx.close()
    await browser.close()

async def main():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["url","label"])
    # Tuple-ize
    items = [(u, int(l)) for u,l in zip(df["url"], df["label"])]
    # Chunk for workers
    chunks = [items[i::CONCURRENCY] for i in range(CONCURRENCY)]
    async with async_playwright() as play:
        await asyncio.gather(*[worker(play, chunk) for chunk in chunks])

if __name__ == "__main__":
    asyncio.run(main())
