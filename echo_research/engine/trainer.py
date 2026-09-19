from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..losses import BoundaryDiceLoss, LVFunctionalConsistencyLoss, SegmentationLoss
from ..metrics.segmentation import segmentation_metrics
from .checkpoint import save_checkpoint, save_json


class Trainer:
    def __init__(self, model, config: Dict, train_loader, val_loader, device: torch.device, run_dir: str | Path):
        self.model = model.to(device)
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tcfg = config["training"]
        lcfg = config["loss"]
        self.task = config["data"]["task"]
        self.num_classes = int(config["model"].get("num_classes", 4))
        self.paired = bool(config["data"].get("paired_phases", True))

        self.seg_loss = SegmentationLoss(self.task, self.num_classes, lcfg.get("ce_weight", 1.0), lcfg.get("dice_weight", 1.0))
        self.boundary_loss = BoundaryDiceLoss(self.task, self.num_classes, lcfg.get("boundary_tolerance_px", 2))
        self.functional_loss = LVFunctionalConsistencyLoss(self.task, lv_class=lcfg.get("lv_class", 1), physiology_weight=lcfg.get("physiology_weight", 0.25))
        self.adv_bce = nn.BCEWithLogitsLoss()

        self.gen_opt = torch.optim.Adam(self.model.generator.parameters(), lr=float(tcfg["lr_g"]), betas=tuple(tcfg.get("betas", [0.5, 0.999])))
        self.disc_opt = torch.optim.Adam(self.model.discriminator.parameters(), lr=float(tcfg["lr_d"]), betas=tuple(tcfg.get("betas", [0.5, 0.999])))
        self.gen_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.gen_opt, mode="max", factor=float(tcfg.get("lr_factor", 0.5)), patience=int(tcfg.get("lr_patience", 10)), min_lr=float(tcfg.get("min_lr", 1e-6)))
        self.disc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.disc_opt, mode="max", factor=float(tcfg.get("lr_factor", 0.5)), patience=int(tcfg.get("lr_patience", 10)), min_lr=float(tcfg.get("min_lr", 1e-6)))
        amp_enabled = bool(tcfg.get("amp", True) and device.type == "cuda")
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.history = []

    def _disc_labels(self, out: torch.Tensor, real: bool) -> torch.Tensor:
        smoothing = float(self.cfg["training"].get("label_smoothing", 0.0))
        if real:
            val = 1.0 - smoothing
        else:
            val = 0.0
        labels = torch.full_like(out, val)
        flip_prob = float(self.cfg["training"].get("label_flip_prob", 0.0))
        if self.model.training and flip_prob > 0:
            flip = torch.rand_like(labels) < flip_prob
            labels = torch.where(flip, 1.0 - labels, labels)
        return labels

    def _one_phase_generator_loss(self, image, target, logits):
        lcfg = self.cfg["loss"]
        total = torch.zeros((), device=self.device)
        details = {}
        seg, seg_parts = self.seg_loss(logits, target)
        total = total + float(lcfg.get("region_weight", 1.0)) * seg
        details.update(seg_parts)

        if float(lcfg.get("boundary_weight", 0.0)) > 0:
            b = self.boundary_loss(logits, target)
            total = total + float(lcfg["boundary_weight"]) * b
            details["boundary_loss"] = b.detach()

        if float(lcfg.get("pixel_l1_weight", 0.0)) > 0:
            fake = self.model.mask_representation(logits, is_logits=True)
            real = self.model.mask_representation(target, is_logits=False)
            pix = F.l1_loss(fake, real)
            total = total + float(lcfg["pixel_l1_weight"]) * pix
            details["pixel_l1"] = pix.detach()

        if float(lcfg.get("adversarial_weight", 0.0)) > 0:
            fake = self.model.mask_representation(logits, is_logits=True)
            d_fake = self.model.discriminator(image, fake)
            adv = self.adv_bce(d_fake, self._disc_labels(d_fake, True))
            total = total + float(lcfg["adversarial_weight"]) * adv
            details["adversarial_loss"] = adv.detach()
        return total, details

    def _discriminator_loss_phase(self, image, target, logits):
        real_mask = self.model.mask_representation(target, is_logits=False)
        fake_mask = self.model.mask_representation(logits.detach(), is_logits=True)
        out_real = self.model.discriminator(image, real_mask)
        out_fake = self.model.discriminator(image, fake_mask)
        loss_real = self.adv_bce(out_real, self._disc_labels(out_real, True))
        loss_fake = self.adv_bce(out_fake, self._disc_labels(out_fake, False))
        return 0.5 * (loss_real + loss_fake)

    def _move(self, batch):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_epoch(self, epoch: int):
        self.model.train()
        sums = defaultdict(float)
        count = 0
        max_grad = float(self.cfg["training"].get("grad_clip", 5.0))
        d_min_loss = float(self.cfg["training"].get("d_update_min_loss", -1.0))
        for batch in self.train_loader:
            batch = self._move(batch)
            if self.paired:
                image_ed, target_ed = batch["image_ed"], batch["mask_ed"]
                image_es, target_es = batch["image_es"], batch["mask_es"]
            else:
                image_ed, target_ed = batch["image"], batch["mask"]
                image_es = target_es = None

            # Generator step. Discriminator weights are frozen to avoid accumulating D gradients.
            for p in self.model.discriminator.parameters():
                p.requires_grad_(False)
            self.gen_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                logits_ed = self.model.generator(image_ed)
                g_loss, parts = self._one_phase_generator_loss(image_ed, target_ed, logits_ed)
                if self.paired:
                    logits_es = self.model.generator(image_es)
                    g2, parts2 = self._one_phase_generator_loss(image_es, target_es, logits_es)
                    g_loss = 0.5 * (g_loss + g2)
                    for k, v in parts2.items():
                        parts[k] = 0.5 * (parts.get(k, v) + v)
                    fw = float(self.cfg["loss"].get("functional_weight", 0.0))
                    if fw > 0:
                        fl, fparts = self.functional_loss(logits_ed, logits_es, target_ed, target_es)
                        g_loss = g_loss + fw * fl
                        parts.update(fparts)
                else:
                    logits_es = None
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.unscale_(self.gen_opt)
            torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_grad)
            self.scaler_g.step(self.gen_opt)
            self.scaler_g.update()

            # Discriminator step.
            for p in self.model.discriminator.parameters():
                p.requires_grad_(True)
            self.disc_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                d_loss = self._discriminator_loss_phase(image_ed, target_ed, logits_ed)
                if self.paired:
                    d_loss = 0.5 * (d_loss + self._discriminator_loss_phase(image_es, target_es, logits_es))
            update_d = d_min_loss < 0 or float(d_loss.detach()) >= d_min_loss
            if update_d:
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.disc_opt)
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_grad)
                self.scaler_d.step(self.disc_opt)
                self.scaler_d.update()

            bs = image_ed.shape[0]
            count += bs
            sums["g_loss"] += float(g_loss.detach()) * bs
            sums["d_loss"] += float(d_loss.detach()) * bs
            sums["d_updated"] += float(update_d) * bs
            for k, v in parts.items():
                sums[k] += float(v) * bs
        return {k: v / max(1, count) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        dice_values = []
        losses = []
        for batch in self.val_loader:
            batch = self._move(batch)
            if self.paired:
                phases = [(batch["image_ed"], batch["mask_ed"]), (batch["image_es"], batch["mask_es"])]
            else:
                phases = [(batch["image"], batch["mask"])]
            for image, target in phases:
                logits = self.model.generator(image)
                seg, _ = self.seg_loss(logits, target)
                losses.append(float(seg))
                if self.task == "binary":
                    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
                    gt = (target[:, 0] if target.ndim == 4 else target).cpu().numpy()
                    for p, t in zip(pred, gt):
                        dice_values.append(segmentation_metrics(p, t)["dice"])
                else:
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    gt = target.cpu().numpy()
                    # Validation checkpoint metric is macro foreground Dice.
                    for p, t in zip(pred, gt):
                        per = []
                        for cls in range(1, self.num_classes):
                            per.append(segmentation_metrics(p == cls, t == cls)["dice"])
                        dice_values.append(float(np.mean(per)))
        return {"val_dice": float(np.mean(dice_values)), "val_region_loss": float(np.mean(losses))}

    def fit(self):
        tcfg = self.cfg["training"]
        max_epochs = int(tcfg["epochs"])
        patience = int(tcfg.get("early_stopping_patience", 30))
        min_delta = float(tcfg.get("early_stopping_min_delta", 1e-4))
        best = -math.inf
        no_improve = 0
        start = time.time()
        for epoch in range(1, max_epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate()
            metric = val_stats["val_dice"]
            self.gen_sched.step(metric)
            self.disc_sched.step(metric)
            improved = metric > best + min_delta
            if improved:
                best = metric
                no_improve = 0
                save_checkpoint(self.run_dir / "best.pt", self.model, self.gen_opt, self.disc_opt, epoch, best, self.cfg)
            else:
                no_improve += 1
            row = {
                "epoch": epoch,
                **train_stats,
                **val_stats,
                "lr_g": self.gen_opt.param_groups[0]["lr"],
                "lr_d": self.disc_opt.param_groups[0]["lr"],
                "best_val_dice": best,
                "elapsed_sec": time.time() - start,
            }
            self.history.append(row)
            save_json(self.run_dir / "history.json", self.history)
            print(f"Epoch {epoch:03d} | G {row['g_loss']:.4f} | D {row['d_loss']:.4f} | val Dice {metric:.4f} | best {best:.4f}")
            if no_improve >= patience:
                print(f"Early stopping: validation Dice did not improve for {patience} epochs.")
                break
        save_checkpoint(self.run_dir / "last.pt", self.model, self.gen_opt, self.disc_opt, epoch, best, self.cfg)
        return best
