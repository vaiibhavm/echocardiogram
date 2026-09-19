#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from echo_research.config import load_config
from echo_research.engine.trainer import Trainer
from echo_research.factory import build_loaders, build_model
from echo_research.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="Override data.root without editing YAML")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=None, help="Override experiment.seed; recorded in resolved config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    seed = int(cfg["experiment"].get("seed", 2026))
    seed_everything(seed, bool(cfg["experiment"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = cfg["experiment"].get("name", Path(args.config).stem)
    run_dir = Path(args.run_dir or Path("runs") / name / f"seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    train_loader, val_loader, splits = build_loaders(cfg)
    (run_dir / "split_manifest.json").write_text(json.dumps({"train": splits[0], "val": splits[1], "test": splits[2]}, indent=2), encoding="utf-8")
    model = build_model(cfg)
    n_g = sum(p.numel() for p in model.generator.parameters())
    n_d = sum(p.numel() for p in model.discriminator.parameters())
    print(f"Device: {device} | Generator params: {n_g:,} | Discriminator params: {n_d:,}")
    Trainer(model, cfg, train_loader, val_loader, device, run_dir).fit()
    print(f"Run saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
