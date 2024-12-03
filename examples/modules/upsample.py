import torch
import torch.nn as nn


class Bilinear(torch.nn.Module):
    def __init__(self, size: list[int]):
        super().__init__()
        self.bilinear = nn.Upsample(size=tuple(size), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.bilinear(x)
        return x