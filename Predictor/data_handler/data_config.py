from Predictor.Bases import BaseConfig
from dataclasses import dataclass


@dataclass
class DataConfigAiShell1(BaseConfig):

    raw_file = 'data/data_aishell.tgz'
    data_root = 'data/'

    sample_rate = 16000
    n_mels = 80
    window_size = 400
    augment = False

    collector_path = 'data/collector'
    vocab_path = 'Predictor/vocab.t'
