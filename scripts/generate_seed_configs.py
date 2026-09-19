#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026,2027,2028,2029,2030])
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        c=yaml.safe_load(yaml.safe_dump(cfg))
        c.setdefault("experiment",{})["seed"]=seed
        p=out/f"{Path(args.config).stem}_seed{seed}.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False),encoding="utf-8")
        print(p)
if __name__ == "__main__": main()
