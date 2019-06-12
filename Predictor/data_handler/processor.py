import torchaudio as ta
import torch as t
import numpy as np

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
    lfr_m: int = 4
    lfr_n: int = 3

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

    def low_frame_rate(self, feature):
        # T, D
        feature = build_LFR_features(feature, self.lfr_m, self.lfr_n)
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
        feature = t.Tensor(self.low_frame_rate(feature))
        return feature


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
    #     LFR_inputs_batch.append(np.vstack(LFR_inputs))
    # return LFR_inputs_batch


if __name__ == '__main__':
    parser = AudioParser()
    feature = parser.parse('../../data/data_aishell/wav/train/S0722/BAC009S0722W0494.wav', True)
    print(feature.shape)
