import torch
import torch.nn.functional as F
from torch import nn


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()

        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        feature_map = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            feature_map.append(x)
        x = self.conv_post(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_map


class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS()
        ])

        self.poolings = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, wav, wav_gen):
        wav_ds = []
        wav_gen_ds = []
        wav_fmaps = []
        wav_gen_fmaps = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                wav = self.poolings[i-1](wav)
                wav_gen = self.poolings[i-1](wav_gen)

            wav_d, wav_fmap = d(wav)
            wav_gen_d, wav_gen_fmap = d(wav_gen)
            wav_ds.append(wav_d)
            wav_fmaps.append(wav_fmap)
            wav_gen_ds.append(wav_gen_d)
            wav_gen_fmaps.append(wav_gen_fmap)

        return wav_ds, wav_gen_ds, wav_fmaps, wav_gen_fmaps
