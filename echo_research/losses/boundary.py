from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _soft_boundary(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilation = F.max_pool2d(x, kernel_size, stride=1, padding=pad)
    erosion = -F.max_pool2d(-x, kernel_size, stride=1, padding=pad)
    return (dilation - erosion).clamp(0.0, 1.0)


def _boundary_dice(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 2, eps: float = 1e-6) -> torch.Tensor:
    pb = _soft_boundary(pred)
    tb = _soft_boundary(target)
    if tolerance > 0:
        k = 2 * tolerance + 1
        pb_tol = F.max_pool2d(pb, k, stride=1, padding=tolerance)
        tb_tol = F.max_pool2d(tb, k, stride=1, padding=tolerance)
    else:
        pb_tol, tb_tol = pb, tb
    # Symmetric tolerant boundary agreement.
    precision = (pb * tb_tol).sum((2, 3)) / (pb.sum((2, 3)) + eps)
    recall = (tb * pb_tol).sum((2, 3)) / (tb.sum((2, 3)) + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return 1.0 - f1.mean()


class BoundaryDiceLoss(nn.Module):
    """Differentiable tolerant boundary F1 loss, foreground classes only."""

    def __init__(self, task: str = "binary", num_classes: int = 4, tolerance: int = 2):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.tolerance = tolerance

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            p = torch.sigmoid(logits)
            if target.ndim == 3:
                target = target.unsqueeze(1)
            return _boundary_dice(p, target.float(), self.tolerance)
        p = torch.softmax(logits, dim=1)[:, 1:]
        one_hot = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()[:, 1:]
        losses = [_boundary_dice(p[:, i:i+1], one_hot[:, i:i+1], self.tolerance) for i in range(p.shape[1])]
        return torch.stack(losses).mean()
