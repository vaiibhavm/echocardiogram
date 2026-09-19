#!/usr/bin/env python3
"""Patient-level EDV/ESV/EF experiment using both 2CH and 4CH LV masks.

This script intentionally labels the estimator as experimental. Before a paper uses mL
values, validate this estimator against official/reference CAMUS clinical values or
replace it with the validated challenge implementation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from echo_research.config import load_config
from echo_research.data.camus import CAMUSSampleKey, _paths, load_nifti, resolve_data_dir
from echo_research.data.transforms import EchoTransform
from echo_research.engine.checkpoint import load_checkpoint
from echo_research.factory import build_model, split_ids_from_config
from echo_research.metrics.clinical import estimate_biplane_volume_ml,ejection_fraction

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True); ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--data-root",default=None); ap.add_argument("--output",default="clinical_biplane_experimental.csv")
    args=ap.parse_args(); cfg=load_config(args.config)
    if args.data_root: cfg["data"]["root"]=args.data_root
    if cfg["data"]["task"]!="binary" or cfg["data"].get("structure")!="LVendo": raise SystemExit("Use binary LVendo configuration.")
    _,_,test_ids=split_ids_from_config(cfg); d=cfg["data"]; data_dir=resolve_data_dir(d["root"],d.get("nifti_dir","database_nifti"))
    tfm=EchoTransform(tuple(d.get("image_size",[256,256])),augment=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=build_model(cfg).to(device); load_checkpoint(args.checkpoint,model,map_location=device); model.eval()
    rows=[]
    for pid in test_ids:
        masks_pred={}; masks_gt={}; spacings={}
        ok=True
        for view in ("2CH","4CH"):
            for phase in ("ED","ES"):
                ip,gp=_paths(data_dir,CAMUSSampleKey(pid,view,phase))
                if not (ip.exists() and gp.exists()): ok=False; break
                img,sp=load_nifti(ip); gt,_=load_nifti(gp); gt=(gt==1).astype(np.float32)
                x,_=tfm(img,gt,task="binary"); logits=model.generator(x[None].to(device)); pred=(torch.sigmoid(logits)[0,0].cpu().numpy()>0.5)
                # For clinical geometry, resize prediction back to original image grid before using original spacing.
                pred_back=torch.nn.functional.interpolate(torch.from_numpy(pred.astype(np.float32))[None,None],size=gt.shape,mode="nearest")[0,0].numpy()>0.5
                masks_pred[(view,phase)]=pred_back; masks_gt[(view,phase)]=gt>0.5; spacings[(view,phase)]=sp
            if not ok: break
        if not ok: continue
        vals={}
        for label,src in [("pred",masks_pred),("gt",masks_gt)]:
            edv=estimate_biplane_volume_ml(src[("2CH","ED")],spacings[("2CH","ED")],src[("4CH","ED")],spacings[("4CH","ED")])
            esv=estimate_biplane_volume_ml(src[("2CH","ES")],spacings[("2CH","ES")],src[("4CH","ES")],spacings[("4CH","ES")])
            vals[f"edv_{label}_ml"]=edv; vals[f"esv_{label}_ml"]=esv; vals[f"ef_{label}"]=ejection_fraction(edv,esv)
        rows.append({"patient_id":pid,**vals})
    df=pd.DataFrame(rows)
    if len(df):
        df["edv_abs_error_ml"]=(df.edv_pred_ml-df.edv_gt_ml).abs(); df["esv_abs_error_ml"]=(df.esv_pred_ml-df.esv_gt_ml).abs(); df["ef_abs_error"]=(df.ef_pred-df.ef_gt).abs()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.output,index=False); print(df.describe(include='all').to_string())
    print("WARNING: validate the volume estimator against official/reference clinical values before publication.")
if __name__=="__main__": main()
