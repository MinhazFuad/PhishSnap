#!/usr/bin/env python3
"""
Grab login-like screenshots using a CSV whose label values are the strings:
  - "legitimate"
  - "phishing"

Outputs:
  <out_dir>/legitimate/*.png
  <out_dir>/phishing/*.png
  <out_dir>/manifest.csv  (image_path,url,label)  # label is 'legitimate' or 'phishing'
  <out_dir>/failed_login_grab.txt

Usage:
  python tools/grab_train_login_pages_labelstr.py path/to/url_train.csv images_train --concurrency 6

Requires:
  pip install playwright pandas tqdm
  python -m playwright install chromium
"""
import argparse, asyncio, csv, json
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Tuple
from tqdm.asyncio import tqdm_asyncio
from playwright.async_api import async_playwright, Playwright, Browser, Page, Error as PWError

def safe_name(url: str) -> str:
    import hashlib
    u = url.strip()
    h = hashlib.sha1(u.encode("utf-8")).hexdigest()[:16]
    host = urlparse(u).hostname or "nohost"
    return f"{host}-{h}.png"

def normalize_label(raw) -> str | None:
    """Normalize label to 'legitimate' or 'phishing'. Return None if unknown."""
    if raw is None: return None
    s = str(raw).strip().lower()
    if s in ("phishing","phish","1","malicious","bad"): return "phishing"
    if s in ("legitimate","legit","0","benign","good"): return "legitimate"
    return None

async def is_login_like(page: Page, keywords: List[str]) -> bool:
    try:
        has_pw = await page.evaluate("() => !!document.querySelector('input[type=\"password\"]')")
        if has_pw: return True
        title = (await page.title()) or ""
        url = page.url or ""
        try:
            text = await page.evaluate("() => document.body ? document.body.innerText.slice(0,2000) : ''")
        except Exception:
            try:
                text = await page.evaluate("() => document.body ? document.body.textContent.slice(0,2000) : ''")
            except Exception:
                text = ""
        low = (title + " " + url + " " + text).lower()
        return any(k in low for k in keywords)
    except Exception:
        return False

async def worker(worker_id: int, play: Playwright, rows: List[Tuple[str,str]], out_root: Path, opts, manifest_entries):
    browser: Browser = await play.chromium.launch(headless=opts.headless, args=opts.browser_args)
    context = await browser.new_context(viewport={"width": opts.width, "height": opts.height},
                                        ignore_https_errors=opts.ignore_https)
    page = await context.new_page()
    if opts.user_agent:
        await context.set_extra_http_headers({"user-agent": opts.user_agent})

    keywords = ["login","sign in","signin","sign-in","password","account","authenticate","log in","user id","username","email"]

    for url, label_str in tqdm_asyncio(rows, desc=f"worker-{worker_id}", leave=False):
        out_dir = out_root / ( "phishing" if label_str == "phishing" else "legitimate" )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / safe_name(url)

        if out_fp.exists():
            manifest_entries.append((str(out_fp), url, label_str))
            continue

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=opts.timeout)
            try:
                await page.wait_for_load_state("networkidle", timeout=opts.network_idle_timeout)
            except Exception:
                pass
            await page.wait_for_timeout(opts.settle_time)

            if await is_login_like(page, keywords):
                try:
                    await page.screenshot(path=str(out_fp), full_page=False)
                    manifest_entries.append((str(out_fp), url, label_str))
                except Exception as e:
                    try:
                        await page.wait_for_timeout(500)
                        await page.screenshot(path=str(out_fp), full_page=False)
                        manifest_entries.append((str(out_fp), url, label_str))
                    except Exception as e2:
                        (out_root / "failed_login_grab.txt").open("a", encoding="utf8").write(
                            f"{label_str}\t{url}\tScreenshotError:{repr(e2)}\n"
                        )
        except PWError as e:
            (out_root / "failed_login_grab.txt").open("a", encoding="utf8").write(
                f"{label_str}\t{url}\tPlaywrightError:{repr(e)}\n"
            )
        except Exception as e:
            (out_root / "failed_login_grab.txt").open("a", encoding="utf8").write(
                f"{label_str}\t{url}\tError:{repr(e)}\n"
            )

    await context.close()
    await browser.close()

async def main_async(args):
    csv_path = Path(args.csv)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    import pandas as pd
    df = pd.read_csv(csv_path)

    # auto-detect url/label columns if not provided
    url_col = args.url_col
    label_col = args.label_col
    if url_col is None:
        for c in df.columns:
            if "url" in c.lower() or "link" in c.lower(): url_col = c; break
    if label_col is None:
        for c in df.columns:
            if "label" in c.lower() or c.lower() in ("y","class","target"): label_col = c; break
    if url_col is None or label_col is None:
        raise SystemExit("Couldn't detect url/label columns. Pass --url-col and --label-col.")

    rows, skipped = [], 0
    for _, r in df.iterrows():
        u = r[url_col]
        raw = r[label_col]
        if not isinstance(u, str) or not u.strip(): continue
        u = u.strip()
        if "://" not in u: u = "http://" + u
        lab = normalize_label(raw)
        if lab is None: skipped += 1; continue
        rows.append((u, lab))

    if not rows:
        raise SystemExit(f"No valid rows found. Skipped: {skipped}")

    # distribute rows across workers
    buckets = [[] for _ in range(args.concurrency)]
    for i, item in enumerate(rows): buckets[i % args.concurrency].append(item)

    manifest_entries = []
    async with async_playwright() as play:
        tasks = [asyncio.create_task(worker(i+1, play, bucket, out_root, args, manifest_entries))
                 for i, bucket in enumerate(buckets) if bucket]
        if tasks: await asyncio.gather(*tasks)

    # write manifest (unique by image path)
    seen = set()
    with manifest_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["image_path","url","label"])
        for img, url, lab in manifest_entries:
            if img in seen: continue
            seen.add(img); w.writerow([img, url, lab])

    print("Done. Saved:", manifest_path)

def main():
    parser = argparse.ArgumentParser(description="Grab login screenshots from CSV with string labels 'legitimate'/'phishing'.")
    parser.add_argument("csv", help="CSV of URLs (e.g., data/url_train.csv)")
    parser.add_argument("out_dir", nargs="?", default="images_train", help="Output root dir")
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
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--browser-args", type=json.loads, default=[])
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
