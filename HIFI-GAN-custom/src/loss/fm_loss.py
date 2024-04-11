from torch import nn
import torch


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fmap_real, fmap_gen):
        loss = 0
        for dr, dg in zip(fmap_real, fmap_gen):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return 2 * loss
