import torchaudio as ta
from Predictor.data_handler.loader import load_wav
import torch as t
from Predictor.data_handler.augments import freq_mask
from Predictor.data_handler.augments import time_mask

#
# def tfm_spectro(ad:Audio, sr=16000, to_db_scale=False, n_fft=1024,
#                 ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
#     # We must reshape signal for torchaudio to generate the spectrogram.
#     mel = ta.transforms.MelSpectrogram(sr=ad.sr, n_mels=n_mels, n_fft=n_fft, ws=ws, hop=hop,
#                                     f_min=f_min, f_max=f_max, pad=pad,)(ad.sig.reshape(1, -1))
#     mel = mel.permute(0,2,1) # swap dimension, mostly to look sane to a human.
#     if to_db_scale: mel = ta.transforms.SpectrogramToDB(stype='magnitude', top_db=f_max)(mel)
#     return mel
#

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
