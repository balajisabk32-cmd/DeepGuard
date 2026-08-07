
import os as _os
import sys as _sys

# Scripts live in scripts/ but resolve paths and imports against the
# REPO ROOT, so they behave identically no matter where they are invoked.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import warnings, os, glob, json, random, gc, time; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2
from src.visual.models import LOADERS, SPECS
from src.rppg.analyze import read_frames
from src.rppg import backends
BASE='SDFVD2.0 Extension of Small Scale Deep Fake Video Dataset'
spec=SPECS["effb7"]; net=LOADERS["effb7"]()
MEAN=np.array(spec.mean,np.float32); STD=np.array(spec.std,np.float32); S=spec.input_size
def crop(fr,box):
    x,y,w,h=[float(v) for v in box]; H,W=fr.shape[:2]
    cx,cy=x+w/2,y+h/2; side=max(w,h)*1.3
    x0,y0=int(max(cx-side/2,0)),int(max(cy-side/2,0)); x1,y1=int(min(cx+side/2,W)),int(min(cy+side/2,H))
    if x1-x0<16 or y1-y0<16: return None
    c=fr[y0:y1,x0:x1]
    if c.size==0: return None
    rgb=cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),(S,S),interpolation=cv2.INTER_AREA)
    return ((rgb.astype(np.float32)/255.0-MEAN)/STD).transpose(2,0,1)
random.seed(0)   # SAME seed as the feature run -> same clips, so scores align
det=backends.OpenCVBackend()
out={}; t0=time.time()
for cls,label in [("SDFVD2.0_fake",1),("SDFVD2.0_real",0)]:
    files=sorted(glob.glob(os.path.join(BASE,cls,"*.mp4")))
    sel=random.sample(files, min(150,len(files)))
    for i,f in enumerate(sel):
        try:
            frames,t,_,_=read_frames(f, max_sec=6.0)
            if len(frames)<12: continue
            boxes,_=det.detect_boxes(frames); boxes=backends.smooth_boxes(boxes)
            idx=[j for j in np.linspace(0,len(frames)-1,8).astype(int) if not np.isnan(boxes[j]).any()]
            ts=[c for c in (crop(frames[j],boxes[j]) for j in idx) if c is not None]
            del frames; gc.collect()
            if not ts: continue
            with torch.no_grad():
                p=torch.sigmoid(net(torch.from_numpy(np.stack(ts))).reshape(-1)).numpy()
            out[os.path.basename(f)]={"score":float(np.percentile(p,90)),"label":label}
        except Exception:
            continue
        if (i+1)%25==0: print(f"  {cls} {i+1} ({time.time()-t0:.0f}s, kept {len(out)})", flush=True)
json.dump(out, open("results/sdfvd_effb7.json","w"), indent=1)
fk=[v["score"] for v in out.values() if v["label"]==1]; rl=[v["score"] for v in out.values() if v["label"]==0]
auc=sum((1.0 if a>b else 0.5 if a==b else 0.0) for a in fk for b in rl)/(len(fk)*len(rl))
print(f"\neffb7 on SDFVD2.0: {len(fk)} fake / {len(rl)} real -> AUC {auc:.3f}  ({time.time()-t0:.0f}s)")
