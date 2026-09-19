from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from ..metrics.segmentation import segmentation_metrics
from .checkpoint import load_checkpoint


@torch.no_grad()
def evaluate_checkpoint(model, loader, checkpoint: str | Path, device: torch.device, task: str, num_classes: int, output_csv: str | Path):
    load_checkpoint(checkpoint, model, map_location=device)
    model.to(device).eval()
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"]
        logits = model.generator(image)
        if task == "binary":
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
            gt = (target[:, 0] if target.ndim == 4 else target).numpy()
            classes = [("foreground", 1)]
        else:
            labels = torch.argmax(logits, dim=1).cpu().numpy()
            gt_labels = target.numpy()
            classes = [(f"class_{c}", c) for c in range(1, num_classes)]
        for i in range(image.shape[0]):
            spacing = tuple(float(x) for x in batch["spacing"][i].numpy())
            base = {
                "patient_id": batch["patient_id"][i],
                "view": batch["view"][i],
                "phase": batch["phase"][i],
            }
            if task == "binary":
                m = segmentation_metrics(pred[i], gt[i], spacing)
                rows.append({**base, "class": "foreground", **m})
            else:
                for name, c in classes:
                    m = segmentation_metrics(labels[i] == c, gt_labels[i] == c, spacing)
                    rows.append({**base, "class": name, **m})
    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary = df.groupby(["view", "phase", "class"])[["dice", "iou", "precision", "recall", "hd95", "asd"]].agg(["mean", "std"])
    summary.to_csv(Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv"))
    return df, summary
