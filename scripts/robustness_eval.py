#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from echo_research.config import load_config
from echo_research.engine.checkpoint import load_checkpoint
from echo_research.factory import build_model, build_test_loader
from echo_research.metrics.segmentation import segmentation_metrics
from echo_research.robustness.corruptions import apply_corruption

KINDS=["gaussian_noise","speckle","blur","low_contrast","shadow"]
SEVERITIES=[0.25,0.5,0.75,1.0]

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--data-root",default=None)
    ap.add_argument("--output",default="robustness_metrics.csv")
    args=ap.parse_args()
    cfg=load_config(args.config)
    if args.data_root: cfg["data"]["root"]=args.data_root
    if cfg["data"]["task"]!="binary":
        raise SystemExit("Current robustness script targets the primary binary LVendo paper track. Extend deliberately for multiclass.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(cfg).to(device)
    load_checkpoint(args.checkpoint,model,map_location=device)
    model.eval(); loader=build_test_loader(cfg)
    rng=np.random.default_rng(2026); rows=[]
    for batch in loader:
        gt=(batch["mask"][:,0] if batch["mask"].ndim==4 else batch["mask"]).numpy()
        for kind in ["clean"]+KINDS:
            sevs=[0.0] if kind=="clean" else SEVERITIES
            for sev in sevs:
                arr=batch["image"].numpy()[:,0]
                corr=np.stack([im if kind=="clean" else apply_corruption(im,kind,sev,rng) for im in arr])
                x=torch.from_numpy(corr[:,None]).float().to(device)
                pred=(torch.sigmoid(model.generator(x))>0.5).cpu().numpy()[:,0]
                for i,(p,t) in enumerate(zip(pred,gt)):
                    spacing=tuple(float(v) for v in batch["spacing"][i].numpy())
                    m=segmentation_metrics(p,t,spacing)
                    rows.append({"patient_id":batch["patient_id"][i],"view":batch["view"][i],"phase":batch["phase"][i],"corruption":kind,"severity":sev,**m})
    df=pd.DataFrame(rows); Path(args.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.output,index=False)
    print(df.groupby(["corruption","severity"])[["dice","hd95","asd"]].mean().to_string())
if __name__=="__main__": main()
