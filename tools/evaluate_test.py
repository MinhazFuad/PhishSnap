#!/usr/bin/env python3
# pip install pillow numpy opencv-python pandas
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import cv2

HASH_SIZE=8; HIGHFREQ=4

def phash_path(p):
    im = Image.open(p).convert("L").resize((HASH_SIZE*HIGHFREQ,)*2, Image.LANCZOS)
    dct = cv2.dct(np.asarray(im, dtype=np.float32))
    d = dct[:HASH_SIZE,:HASH_SIZE]
    med = np.median(d[1:,1:])
    bits = (d>med).astype(np.uint8).flatten()
    v=0
    for b in bits: v=(v<<1)|int(b)
    return int(v)

def popcount64(x):
    c=0
    while x:
        x &= (x-1); c+=1
    return c

def to01(lbl):
    s=str(lbl).strip().lower()
    return 1 if s in ("phish","phishing","1") else 0

def nearest(train, h):
    best_d=999; best_lbl=None
    for t in train:
        d = popcount64(h ^ t["hash"])
        if d < best_d:
            best_d, best_lbl = d, t["label"]
            if best_d<=2: break
    return best_lbl, best_d

def metrics(y_true, y_pred):
    tp=sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==1)
    tn=sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==0)
    fp=sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==1)
    fn=sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==0)
    prec= tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1  = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0.0
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,precision=prec,recall=rec,f1=f1,accuracy=acc)

def main(index_json, test_manifest, thr):
    train = json.loads(Path(index_json).read_text())
    test = pd.read_csv(test_manifest)
    rows=[]
    for _,r in test.iterrows():
        p=Path(r["image_path"])
        if not p.exists(): continue
        h=phash_path(p)
        true=to01(r["label"])
        nn_lbl, dist = nearest(train, h)
        pred = 1 if (nn_lbl==1 and dist<=thr) else 0
        rows.append((true,pred))
    if not rows:
        print("No test images found."); return
    y_true=[a for a,b in rows]; y_pred=[b for a,b in rows]
    print("Test metrics @ threshold", thr, ":", metrics(y_true,y_pred))

if __name__=="__main__":
    idx  = sys.argv[1] if len(sys.argv)>1 else "phash_index_train.json"
    mani = sys.argv[2] if len(sys.argv)>2 else "images_test/manifest.csv"
    thr  = int(sys.argv[3]) if len(sys.argv)>3 else 11
    main(idx, mani, thr)
