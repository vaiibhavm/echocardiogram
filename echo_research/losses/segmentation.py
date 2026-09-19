from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, include_background: bool = False, eps: float = 1e-6) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    start = 0 if include_background else 1
    p = p[:, start:]
    one_hot = one_hot[:, start:]
    dims = (0, 2, 3)
    inter = (p * one_hot).sum(dims)
    denom = p.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


class SegmentationLoss(nn.Module):
    """Region loss: BCE/CE + soft Dice."""

    def __init__(self, task: str = "binary", num_classes: int = 4, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if self.task == "binary":
            if target.ndim == 3:
                target = target.unsqueeze(1)
            ce = F.binary_cross_entropy_with_logits(logits, target.float())
            dice = binary_dice_loss(logits, target.float())
        else:
            ce = F.cross_entropy(logits, target.long())
            dice = multiclass_dice_loss(logits, target, self.num_classes)
        total = self.ce_weight * ce + self.dice_weight * dice
        return total, {"region_ce": ce.detach(), "region_dice_loss": dice.detach()}
