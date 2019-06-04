import torchaudio as ta
import torch as t

from Predictor.data_handler import load_wav
from Predictor.data_handler import freq_mask
from Predictor.data_handler import time_mask


class AudioParser:

    def __init__(self, sr=16000, n_mels=40, n_fft=400, ws=None, hop=None, f_min=0.0, f_max=-80, pad=0):
        self.transform_fn = ta.transforms.MelSpectrogram(
            sr=sr, n_mels=n_mels, n_fft=n_fft, ws=ws, hop=hop, f_min=f_min, f_max=f_max, pad=pad
        )

    def transform(self, signal: t.Tensor, sample_rate: int) -> t.Tensor:
        assert self.transform_fn.sr == sample_rate
        feature = self.transform_fn(signal).squeeze(0)
        return feature

    def normalize(self, feature: t.Tensor):
        feature = (feature - feature.mean()) / feature.std()
        return feature

    def augment(self, feature):
        feature.transpose_(-1, -2)
        feature = time_mask(feature)
        feature = freq_mask(feature)
        feature.transpose_(-1, -2)
        #feature = self.time_shift(feature)
        return feature

    def parse(self, path: str, augment: bool=False) -> t.Tensor:
        signal, sample_rate = load_wav(path)
        feature = self.transform(signal, sample_rate)
        feature = self.normalize(feature)
        if augment:
            feature = self.augment(feature)
        else:
            pass
        return feature


if __name__ == '__main__':
    parser = AudioParser()
    feature = parser.parse('../../data/data_aishell/wav/train/S0722/BAC009S0722W0494.wav')
    print(feature.shape)
