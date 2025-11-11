#!/usr/bin/env python3
"""
build_phash_from_manifest.py
--------------------------------
Reads a manifest.csv (image_path, url, label)
→ computes perceptual hashes (pHash) for each image
→ saves them to a JSON file (default: phash_index_train.json)

Usage:
  python tools/build_phash_from_manifest.py images_train/manifest.csv phash_index_train.json

Requires:
  pip install pillow numpy opencv-python pandas
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

HASH_SIZE = 8
HIGHFREQ_FACTOR = 4

def compute_phash(image_path: Path) -> int:
    """Compute an 8x8 perceptual hash (64-bit integer)."""
    im = Image.open(image_path).convert("L").resize((HASH_SIZE*HIGHFREQ_FACTOR,)*2, Image.LANCZOS)
    pixels = np.asarray(im, dtype=np.float32)
    dct = cv2.dct(pixels)
    dctlow = dct[:HASH_SIZE, :HASH_SIZE]
    med = np.median(dctlow[1:, 1:])  # ignore DC coefficient
    bits = (dctlow > med).astype(np.uint8).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return int(value)

def normalize_label(label):
    """Convert label strings to numeric: legitimate→0, phishing→1."""
    s = str(label).strip().lower()
    if s in ("phish", "phishing", "1"): return 1
    if s in ("legit", "legitimate", "0"): return 0
    return None

def main(manifest_csv, out_json="phash_index_train.json"):
    df = pd.read_csv(manifest_csv)
    expected_cols = {"image_path", "label"}
    if not expected_cols.issubset(set(df.columns)):
        raise SystemExit(f"manifest.csv must have at least columns: {expected_cols}, found: {df.columns.tolist()}")

    results = []
    for i, row in df.iterrows():
        path = Path(row["image_path"])
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue
        label = normalize_label(row["label"])
        if label is None:
            print(f"[skip] unknown label: {row['label']}")
            continue
        try:
            h = compute_phash(path)
            results.append({"hash": h, "label": label, "path": str(path)})
        except Exception as e:
            print(f"[error] {path}: {e}")

    Path(out_json).write_text(json.dumps(results))
    print(f"✅ Wrote {len(results)} entries to {out_json}")

if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else "images_train/manifest.csv"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "phash_index_train.json"
    main(manifest, out_json)
