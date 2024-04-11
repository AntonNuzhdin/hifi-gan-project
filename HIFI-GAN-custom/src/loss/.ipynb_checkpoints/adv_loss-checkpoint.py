import torch
from torch import nn


class GeneratorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_outputs):
        adv_loss = 0.0

        for pred_output in d_outputs:
            pred_loss = torch.mean((pred_output - 1) ** 2)
            adv_loss += pred_loss
        return adv_loss


class DiscriminatorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_outputs, pred_outputs):
        adv_loss = 0.0
        for gt_output, pred_output in zip(gt_outputs, pred_outputs):
            gt_loss = torch.mean((gt_output - 1) ** 2)
            pred_loss = torch.mean(pred_output ** 2)
            adv_loss += gt_loss + pred_loss
        return adv_loss
