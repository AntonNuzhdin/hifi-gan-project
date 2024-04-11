from torch import nn
from src.loss.adv_loss import DiscriminatorAdvLoss, GeneratorAdvLoss
from src.loss.fm_loss import FeatureMatchingLoss
from src.loss.mel_loss import MelLoss


class HIFIGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_adv_loss = DiscriminatorAdvLoss()
        self.gen_adv_loss = GeneratorAdvLoss()
        self.fm_loss = FeatureMatchingLoss()
        self.mel_loss = MelLoss()
