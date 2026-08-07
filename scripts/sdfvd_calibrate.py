
import os as _os
import sys as _sys

# Scripts live in scripts/ but resolve paths and imports against the
# REPO ROOT, so they behave identically no matter where they are invoked.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import warnings, os, glob, json, random, gc, time; warnings.filterwarnings("ignore")
import numpy as np
from src.visual.pixel_forensics import extract_features
from src.rppg.analyze import read_frames
from src.rppg import backends

BASE='SDFVD2.0 Extension of Small Scale Deep Fake Video Dataset'
N_PER_CLASS=int(os.environ.get("N_PER_CLASS","120"))
random.seed(0)
det=backends.OpenCVBackend()
rows=[]; t0=time.time()
for cls,label in [("SDFVD2.0_fake",1),("SDFVD2.0_real",0)]:
    files=sorted(glob.glob(os.path.join(BASE,cls,"*.mp4")))
    for i,f in enumerate(random.sample(files, min(N_PER_CLASS,len(files)))):
        try:
            frames,t,_,_=read_frames(f, max_sec=6.0)
            if len(frames)<12: continue
            boxes,_=det.detect_boxes(frames); boxes=backends.smooth_boxes(boxes)
            feats=extract_features(frames, boxes, max_frames=24)
            del frames; gc.collect()
            if not feats: continue
            feats["__label"]=label; feats["__file"]=os.path.basename(f)
            rows.append(feats)
        except Exception:
            continue
        if (i+1)%20==0: print(f"  {cls} {i+1}  ({time.time()-t0:.0f}s, kept {len(rows)})", flush=True)
json.dump(rows, open("results/sdfvd_features.json","w"))
print(f"extracted {len(rows)} clips in {time.time()-t0:.0f}s -> results/sdfvd_features.json")
