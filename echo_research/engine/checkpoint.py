from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, model, gen_opt, disc_opt, epoch: int, best_metric: float, config: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_metric": best_metric,
        "generator": model.generator.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "gen_optimizer": gen_opt.state_dict() if gen_opt else None,
        "disc_optimizer": disc_opt.state_dict() if disc_opt else None,
        "config": config,
    }, path)


def load_checkpoint(path: str | Path, model, gen_opt=None, disc_opt=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.generator.load_state_dict(ckpt["generator"])
    model.discriminator.load_state_dict(ckpt["discriminator"])
    if gen_opt is not None and ckpt.get("gen_optimizer"):
        gen_opt.load_state_dict(ckpt["gen_optimizer"])
    if disc_opt is not None and ckpt.get("disc_optimizer"):
        disc_opt.load_state_dict(ckpt["disc_optimizer"])
    return ckpt


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
