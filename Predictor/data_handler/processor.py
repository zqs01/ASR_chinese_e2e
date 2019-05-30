import torchaudio as ta
from Predictor.data_handler.loader import load_wav
import torch as t
from .sparse_image_warp import sparse_image_warp

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

    def __init__(self, sr=16000, to_db_scale=False, n_fft=1024, ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
        #self.transform_fn = ta.transforms.MFCC(sample_rate, n_mfcc, dct_type, norm, log_mels, melkwargs)
        self.transform_fn = ta.transforms.MelSpectrogram(
            sr=sr, n_mels=n_mels, n_fft=n_fft, ws=ws, hop=hop, f_min=f_min, f_max=f_max, pad=pad
        )
    def transform(self, signal: t.Tensor, sample_rate: int) -> t.Tensor:
        assert self.transform_fn.sr == sample_rate
        feature = self.transform_fn(signal).squeeze(0)
        return feature

    def normalize(self, feature: t.Tensor):
        mean = feature.mean(-1, keepdim=True)
        std = feature.std(-1, keepdim=True)
        feature.add_(-mean)
        feature.div_(std)
        return feature

    def argument(self, feature):
        feature = self.time_mask(feature)
        feature = self.freq_mask(feature)
        feature = self.time_shift(feature)
        return feature

    def time_mask(self, feature):
        pass

    def freq_mask(self, feature):
        pass

    def time_shift(self, feature):
        pass

    def parse(self, path: str, argument: bool=False) -> t.Tensor:
        signal, sample_rate = load_wav(path)
        feature = self.transform(signal, sample_rate)
        feature = self.normalize(feature)
        return feature


if __name__ == '__main__':
    parser = AudioParser()
    feature = parser.parse('data/ASR_chinese_e2e/data/data_aishell/wav/train/S0722/BAC009S0722W0494.wav')
    print(feature.shape)
