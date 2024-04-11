import random
import torch
import librosa
import os

from pathlib import Path
from librosa.util import normalize
from torch.utils.data import Dataset
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(Dataset):
    def __init__(self, dataset_path, split_path, wav_max_len=None, items_limit=None) -> None:
        super().__init__()

        self.split_path = split_path
        self.dataset_path = dataset_path

        self.wav_paths = []
        dataset_path = Path(dataset_path)

        for wav_path in self._get_dataset_filelist():
            self.wav_paths.append(wav_path)

        if items_limit is not None:
            self.wav_paths = self.wav_paths[:items_limit]

        mel_config = MelSpectrogramConfig()
        self.convert_to_mel = MelSpectrogram(mel_config)
        self.wav_max_len = wav_max_len

    def __len__(self):
        return len(self.wav_paths)

    def _get_dataset_filelist(self):
        with open(self.split_path, 'r', encoding='utf-8') as fi:
            files = [os.path.join(self.dataset_path, x.split('|')[0] + '.wav')
                     for x in fi.read().split('\n') if len(x) > 0]
        return files

    def __getitem__(self, index):
        wav, _ = librosa.load(self.wav_paths[index])
        wav = torch.FloatTensor(librosa.util.normalize(wav)) * 0.95
        if self.wav_max_len is not None:
            start = random.randint(0, wav.shape[-1] - self.wav_max_len)
            wav = wav[..., start: start + self.wav_max_len]
        wav = torch.nn.functional.pad(wav, (0, self.wav_max_len - wav.shape[0]), 'constant')
        mel = self.convert_to_mel(wav.detach())
        return {'wav': wav.squeeze(0), 'mel': mel.squeeze(0)}
