import torchaudio as ta
import torch as t


def load_wav(path: str) -> (t.Tensor, int):
    """
    load wav file and do mean for multi channel
    :param path:
    :return: t
    """
    tensor, sample_rate = ta.load(path, normalization=True)
    num_channel, length = tensor.size()
    if num_channel == 1:
        tensor = tensor
    else:
        tensor = tensor.mean(dim=0, keep_dim=True)
    return tensor, sample_rate



