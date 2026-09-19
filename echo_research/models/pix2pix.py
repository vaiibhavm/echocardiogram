from __future__ import annotations

import torch
from torch import nn

from .patchgan import PatchDiscriminator
from .unet import UNetGenerator


class Pix2PixResearchModel(nn.Module):
    def __init__(self, task: str = "binary", num_classes: int = 4, base_channels: int = 64):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        out_channels = 1 if task == "binary" else num_classes
        mask_channels = out_channels
        self.generator = UNetGenerator(1, out_channels, base_channels)
        self.discriminator = PatchDiscriminator(1, mask_channels, base_channels)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits) if self.task == "binary" else torch.softmax(logits, dim=1)

    def mask_representation(self, logits_or_mask: torch.Tensor, is_logits: bool) -> torch.Tensor:
        if self.task == "binary":
            if is_logits:
                return torch.sigmoid(logits_or_mask)
            if logits_or_mask.ndim == 3:
                logits_or_mask = logits_or_mask.unsqueeze(1)
            return logits_or_mask.float()
        if is_logits:
            return torch.softmax(logits_or_mask, dim=1)
        if logits_or_mask.ndim == 4 and logits_or_mask.shape[1] == self.num_classes:
            return logits_or_mask.float()
        return torch.nn.functional.one_hot(logits_or_mask.long(), self.num_classes).permute(0, 3, 1, 2).float()
