from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LVFunctionalConsistencyLoss(nn.Module):
    """ED/ES functional-consistency loss based on soft LV area change.

    This is deliberately NOT called an EF loss. 2D area change is a differentiable
    surrogate used during training. Publication-grade EF/EDV/ESV must be evaluated
    separately with a validated biplane volume protocol.
    """

    def __init__(self, task: str = "binary", lv_class: int = 1, physiology_weight: float = 0.25, eps: float = 1e-6):
        super().__init__()
        self.task = task
        self.lv_class = lv_class
        self.physiology_weight = physiology_weight
        self.eps = eps

    def _prob(self, logits: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            return torch.sigmoid(logits)[:, 0]
        return torch.softmax(logits, dim=1)[:, self.lv_class]

    def _target(self, target: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            return target[:, 0] if target.ndim == 4 else target.float()
        return (target == self.lv_class).float()

    def forward(self, logits_ed: torch.Tensor, logits_es: torch.Tensor, target_ed: torch.Tensor, target_es: torch.Tensor):
        ped, pes = self._prob(logits_ed), self._prob(logits_es)
        ted, tes = self._target(target_ed), self._target(target_es)
        aed_p = ped.mean((1, 2))
        aes_p = pes.mean((1, 2))
        aed_t = ted.mean((1, 2))
        aes_t = tes.mean((1, 2))
        frac_p = (aed_p - aes_p) / (aed_p + self.eps)
        frac_t = (aed_t - aes_t) / (aed_t + self.eps)
        consistency = F.smooth_l1_loss(frac_p, frac_t)
        physiology = torch.relu(aes_p - aed_p).mean()
        total = consistency + self.physiology_weight * physiology
        return total, {
            "functional_area_change": consistency.detach(),
            "functional_physiology": physiology.detach(),
        }
