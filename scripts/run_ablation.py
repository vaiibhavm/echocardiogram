#!/usr/bin/env python3
"""Print or execute the recommended ablation matrix.

For publication experiments, run each config with seeds 2026, 2027, 2028, 2029, 2030.
This script deliberately requires --execute before launching expensive jobs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    "configs/ablation/a0_controlled_pix2pix.yaml",
    "configs/ablation/a1_plus_dice.yaml",
    "configs/ablation/a2_plus_boundary.yaml",
    "configs/ablation/a3_plus_functional.yaml",
    "configs/proposed_binary.yaml",
]
SEEDS = [2026, 2027, 2028, 2029, 2030]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    for cfg_rel in CONFIGS:
        for seed in SEEDS:
            cfg = root / cfg_rel
            cmd = [sys.executable, str(root / "scripts/train.py"), "--config", str(cfg), "--data-root", args.data_root,
                   "--run-dir", str(root / "runs" / Path(cfg_rel).stem / f"seed_{seed}"), "--seed", str(seed)]
            print(" ".join(cmd))
            if args.execute:
                subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
