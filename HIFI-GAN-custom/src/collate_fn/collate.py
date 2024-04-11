import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def pad_wav(wavs, padding_value=0):
    return pad_sequence([element['wav'] for element in wavs], padding_value=padding_value, batch_first=True)


def pad_mels(mel, max_len):
    return F.pad(mel, (0, max_len - mel.size(-1), 0, 0))


def collate_fn(batch):
    padded_wavs = pad_wav(batch).unsqueeze(1)

    max_length = max([x['mel'].size(-1) for x in batch])
    padded_mels = torch.stack([pad_mels(x['mel'], max_length) for x in batch])

    output = {
        'wav': padded_wavs,
        'mel': padded_mels
    }
    return output
