#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from echo_research.config import load_config
from echo_research.engine.evaluator import evaluate_checkpoint
from echo_research.factory import build_model, build_test_loader


def main():
    ap = argparse.ArgumentParser(description="Final TEST evaluation. Do not use this for model selection.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output", default="test_metrics.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    loader = build_test_loader(cfg)
    df, summary = evaluate_checkpoint(model, loader, args.checkpoint, device, cfg["data"]["task"], int(cfg["model"].get("num_classes", 4)), args.output)
    print(summary)
    print(f"Per-sample metrics: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
