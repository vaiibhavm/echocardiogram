from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.camus import CAMUSFrameDataset, CAMUSPhasePairDataset, load_split_ids, validate_split_safety
from .models.pix2pix import Pix2PixResearchModel


def split_ids_from_config(cfg):
    d = cfg["data"]
    common = dict(
        root=d["root"],
        nifti_dir=d.get("nifti_dir", "database_nifti"),
        split_dir=d.get("split_dir", "database_split"),
        train_file=d.get("train_split_file", "subgroup_training.txt"),
        val_file=d.get("val_split_file", "subgroup_validation.txt"),
        test_file=d.get("test_split_file", "subgroup_testing.txt"),
        val_fraction=float(d.get("val_fraction", 0.15)),
        seed=int(cfg["experiment"].get("seed", 2026)),
    )
    tr = load_split_ids(split="train", **common)
    va = load_split_ids(split="val", **common)
    te = load_split_ids(split="test", **common)
    validate_split_safety(tr, va, te)
    return tr, va, te


def _dataset(cfg, ids, split, paired=None):
    d = cfg["data"]
    paired = d.get("paired_phases", True) if paired is None else paired
    cls = CAMUSPhasePairDataset if paired else CAMUSFrameDataset
    kwargs = dict(
        root=d["root"], patient_ids=ids, views=d.get("views", ["2CH", "4CH"]),
        task=d.get("task", "binary"), structure=d.get("structure", "LVendo"),
        image_size=tuple(d.get("image_size", [256, 256])), augment=(split == "train" and bool(d.get("augment", True))),
        augment_cfg=d.get("augmentation", {}), nifti_dir=d.get("nifti_dir", "database_nifti"),
    )
    if not paired:
        kwargs["phases"] = d.get("phases", ["ED", "ES"])
    return cls(**kwargs)


def build_loaders(cfg):
    tr, va, te = split_ids_from_config(cfg)
    tcfg = cfg["training"]
    workers = int(tcfg.get("num_workers", 4))
    common = dict(batch_size=int(tcfg.get("batch_size", 2)), num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=(workers > 0))
    train_ds = _dataset(cfg, tr, "train")
    val_ds = _dataset(cfg, va, "val")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, (tr, va, te)


def build_test_loader(cfg):
    _, _, te = split_ids_from_config(cfg)
    ds = _dataset(cfg, te, "test", paired=False)
    tcfg = cfg["training"]
    workers = int(tcfg.get("num_workers", 4))
    return DataLoader(ds, batch_size=int(tcfg.get("eval_batch_size", tcfg.get("batch_size", 2))), shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=(workers > 0))


def build_model(cfg):
    return Pix2PixResearchModel(
        task=cfg["data"].get("task", "binary"),
        num_classes=int(cfg["model"].get("num_classes", 4)),
        base_channels=int(cfg["model"].get("base_channels", 64)),
    )
