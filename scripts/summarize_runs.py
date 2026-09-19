#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/**/test_metrics.csv")
    ap.add_argument("--output", default="runs/summary_across_runs.csv")
    args=ap.parse_args()
    files=list(Path('.').glob(args.glob))
    if not files: raise SystemExit(f"No files matched {args.glob}")
    rows=[]
    for f in files:
        df=pd.read_csv(f)
        means=df[["dice","iou","hd95","asd"]].mean(numeric_only=True).to_dict()
        rows.append({"run":str(f.parent),**means})
    out=pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(out.to_string(index=False))
if __name__ == "__main__": main()
