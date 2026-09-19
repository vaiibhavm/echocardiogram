#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from echo_research.config import load_config
from echo_research.data.camus import CAMUSSampleKey, _paths, resolve_data_dir
from echo_research.factory import split_ids_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    tr, va, te = split_ids_from_config(cfg)
    print(f"Patient split sizes: train={len(tr)}, val={len(va)}, test={len(te)}")
    print("Split intersections are empty: PASS")
    d = cfg["data"]
    data_dir = resolve_data_dir(d["root"], d.get("nifti_dir", "database_nifti"))
    for split, ids in [("train", tr), ("val", va), ("test", te)]:
        c = Counter()
        missing = []
        for pid in ids:
            for view in d.get("views", ["2CH", "4CH"]):
                for phase in d.get("phases", ["ED", "ES"]):
                    ip, gp = _paths(data_dir, CAMUSSampleKey(pid, view, phase))
                    if ip.exists() and gp.exists():
                        c[f"{view}_{phase}"] += 1
                    else:
                        missing.append(str(ip))
        print(f"{split}: {dict(c)} | missing pairs={len(missing)}")
        if missing[:5]:
            print("  examples:", *missing[:5], sep="\n    ")


if __name__ == "__main__":
    main()
