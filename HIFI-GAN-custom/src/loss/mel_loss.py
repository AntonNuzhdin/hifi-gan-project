from torch import nn
import torch.nn.functional as F


class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mel_real, mel_gen):
        return 45 * F.l1_loss(mel_real, mel_gen)
