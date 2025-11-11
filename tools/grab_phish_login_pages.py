#!/usr/bin/env python3
"""
tools/grab_phish_login_pages.py

Grab login-like screenshots from a CSV containing only phishing URLs.

Outputs:
  <out_dir>/phishing/*.png      # default out_dir is images_train, so => images_train/phishing/
  <out_dir>/manifest.csv        (image_path,url,label='phishing')
  <out_dir>/failed_login_grab.txt

Usage:
  python tools/grab_phish_login_pages.py data/Phishing_URLs_5k.csv images_train --concurrency 6

Requires:
  pip install playwright pandas tqdm
  python -m playwright install chromium
"""
import argparse
import asyncio
import csv
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import List
from tqdm.asyncio import tqdm_asyncio
from playwright.async_api import async_playwright, Playwright, Browser, Page, Error as PWError

# ---------- Helpers ----------
def safe_name(url: str) -> str:
    import hashlib
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    host = urlparse(url).hostname or "nohost"
    return f"{host}-{h}.png"

async def is_login_like(page: Page, keywords: List[str]) -> bool:
    """Detects if the page looks like a login page."""
    try:
        # Check password input
        has_pw = await page.evaluate("() => !!document.querySelector('input[type=\"password\"]')")
        if has_pw:
            return True
        # Otherwise look for login keywords
        title = (await page.title()) or ""
        url = page.url or ""
        try:
            text = await page.evaluate("() => document.body ? document.body.innerText.slice(0,2000) : ''")
        except:
            text = ""
        low = (title + " " + url + " " + text).lower()
        return any(k in low for k in keywords)
    except Exception:
        return False

# ---------- Worker ----------
async def worker(worker_id: int, play: Playwright, urls: List[str], out_root: Path, opts, manifest_entries):
    browser: Browser = await play.chromium.launch(headless=opts.headless, args=opts.browser_args)
    context = await browser.new_context(viewport={"width": opts.width, "height": opts.height},
                                        ignore_https_errors=opts.ignore_https)
    page = await context.new_page()
    if opts.user_agent:
        await context.set_extra_http_headers({"user-agent": opts.user_agent})

    keywords = ["login","sign in","signin","sign-in","password","account","authenticate","log in","user id","username","email"]
    out_dir = out_root / "phishing"
    out_dir.mkdir(parents=True, exist_ok=True)

    for url in tqdm_asyncio(urls, desc=f"worker-{worker_id}", leave=False):
        img_path = out_dir / safe_name(url)
        if img_path.exists():
            manifest_entries.append((str(img_path), url, "phishing"))
            continue
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=opts.timeout)
            try:
                await page.wait_for_load_state("networkidle", timeout=opts.network_idle_timeout)
            except:
                pass
            await page.wait_for_timeout(opts.settle_time)
            if await is_login_like(page, keywords):
                try:
                    await page.screenshot(path=str(img_path), full_page=False)
                    manifest_entries.append((str(img_path), url, "phishing"))
                except Exception as e:
                    (out_root / "failed_login_grab.txt").open("a", encoding="utf8").write(f"{url}\tScreenshotError:{repr(e)}\n")
        except PWError as e:
            (out_root / "failed_login_grab.txt").open("a", encoding="utf8").write(f"{url}\tPlaywrightError:{repr(e)}\n")
        except Exception as e:
            (out_root / "failed_login_grab.txt").open("a", encoding="utf8").write(f"{url}\tError:{repr(e)}\n")

    await context.close()
    await browser.close()

# ---------- Main ----------
async def main_async(args):
    csv_path = Path(args.csv)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    import pandas as pd
    df = pd.read_csv(csv_path)
    # auto-detect url column
    url_col = args.url_col
    if url_col is None:
        for c in df.columns:
            if "url" in c.lower() or "link" in c.lower():
                url_col = c; break
        if url_col is None:
            url_col = df.columns[0]

    urls = []
    for u in df[url_col]:
        if isinstance(u, str) and u.strip():
            u = u.strip()
            if "://" not in u:
                u = "http://" + u
            urls.append(u)

    if not urls:
        raise SystemExit("No valid URLs found in the CSV.")

    # split across workers
    workers = args.concurrency
    buckets = [[] for _ in range(workers)]
    for i, u in enumerate(urls):
        buckets[i % workers].append(u)

    manifest_entries = []
    async with async_playwright() as play:
        tasks = [asyncio.create_task(worker(i+1, play, bucket, out_root, args, manifest_entries))
                 for i, bucket in enumerate(buckets) if bucket]
        if tasks:
            await asyncio.gather(*tasks)

    # write manifest
    with manifest_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "url", "label"])
        for img, url, lab in manifest_entries:
            w.writerow([img, url, lab])

    print(f"Done. Saved {len(manifest_entries)} screenshots. Manifest: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Grab login screenshots from phishing URLs CSV.")
    parser.add_argument("csv", help="CSV of phishing URLs (e.g., data/Phishing_URLs_5k.csv)")
    parser.add_argument("out_dir", nargs="?", default="images_train", help="Output directory (default: images_train)")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=20000)
    parser.add_argument("--network_idle_timeout", type=int, default=4000)
    parser.add_argument("--settle_time", type=int, default=1200)
    parser.add_argument("--width", type=int, default=1366)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--headless", dest="headless", action="store_true"); parser.set_defaults(headless=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--ignore-https", dest="ignore_https", action="store_true")
    parser.add_argument("--user-agent", type=str, default=None)
    parser.add_argument("--url-col", type=str, default=None)
    parser.add_argument("--browser-args", type=json.loads, default=[])
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
