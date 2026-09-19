from __future__ import annotations

import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNetGenerator(nn.Module):
    """Pix2Pix-style 8-level U-Net for 256x256 inputs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64):
        super().__init__()
        b = base
        self.d1 = DownBlock(in_channels, b, norm=False)
        self.d2 = DownBlock(b, b * 2)
        self.d3 = DownBlock(b * 2, b * 4)
        self.d4 = DownBlock(b * 4, b * 8)
        self.d5 = DownBlock(b * 8, b * 8)
        self.d6 = DownBlock(b * 8, b * 8)
        self.d7 = DownBlock(b * 8, b * 8)
        self.d8 = DownBlock(b * 8, b * 8, norm=False)

        self.u1 = UpBlock(b * 8, b * 8, 0.5)
        self.u2 = UpBlock(b * 16, b * 8, 0.5)
        self.u3 = UpBlock(b * 16, b * 8, 0.5)
        self.u4 = UpBlock(b * 16, b * 8)
        self.u5 = UpBlock(b * 16, b * 4)
        self.u6 = UpBlock(b * 8, b * 2)
        self.u7 = UpBlock(b * 4, b)
        self.final = nn.ConvTranspose2d(b * 2, out_channels, 4, 2, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        u1 = torch.cat([self.u1(d8), d7], dim=1)
        u2 = torch.cat([self.u2(u1), d6], dim=1)
        u3 = torch.cat([self.u3(u2), d5], dim=1)
        u4 = torch.cat([self.u4(u3), d4], dim=1)
        u5 = torch.cat([self.u5(u4), d3], dim=1)
        u6 = torch.cat([self.u6(u5), d2], dim=1)
        u7 = torch.cat([self.u7(u6), d1], dim=1)
        return self.final(u7)
