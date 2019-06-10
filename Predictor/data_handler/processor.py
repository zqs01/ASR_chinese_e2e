import torchaudio as ta
import torch as t

from Predictor.data_handler import load_wav
from Predictor.data_handler.augments import freq_mask
from Predictor.data_handler.augments import time_mask
from dataclasses import dataclass

import seaborn as sns


@dataclass
class AudioParser:
    sample_rate: int = 16000
    n_mels: int = 40
    window_size: int = 400
    hop: int = 160
    f_min: int = 40
    f_max: int = -200
    pad: int = 0

    def load(self, path):
        signal, sample_rate = load_wav(path)
        return signal, sample_rate

    def transform(self, signal: t.Tensor, sample_rate: int) -> t.Tensor:
        assert self.sample_rate == sample_rate
        feature = ta.transforms.MelSpectrogram(
            sr=self.sample_rate, ws=self.window_size, hop=self.hop, f_min=self.f_min, f_max=self.f_max, pad=self.pad,
            n_mels=self.n_mels)(signal)
        feature = t.log(feature.squeeze(0).transpose(0, 1) + 1e-20)
        # faeture : [n_mels, seqlen]
        return feature

    def normalize(self, feature: t.Tensor):
        # faeture : [n_mels, seqlen]
        feature = (feature - feature.mean(1).unsqueeze(1)) / feature.std(1).unsqueeze(1)
        # faeture : [n_mels, seqlen]
        return feature

    def augment(self, feature):
        f = feature.unsqueeze(0)
        feature = time_mask(f)
        feature = freq_mask(feature)
        feature = feature.squeeze(0)
        #feature = self.time_shift(feature)
        return feature

    def parse(self, path: str, augment: bool=False) -> t.Tensor:
        signal, sample_rate = self.load(path)
        feature = self.transform(signal, sample_rate)
        feature = self.normalize(feature)
        if augment:
            feature = self.augment(feature)
        else:
            pass
        feature = feature.transpose(0, 1)
        return feature


if __name__ == '__main__':
    parser = AudioParser()
    feature = parser.parse('../../data/data_aishell/wav/train/S0722/BAC009S0722W0494.wav', True)
    print(feature.shape)
