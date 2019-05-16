import torchaudio as ta
from dataclasses import dataclass
import torch as t
import os
from sklearn.model_selection import train_test_split
from Predictor.Utils.transforms import *


@dataclass
class RawCollector:
    wave_folder = 'data/wav/'
    target_file = 'data/trans.txt'

    def __post_init__(self):
        print(f'collecting...')
        self.load_wave_text_dict()
        self.build_wave_list()

    def load_wave_text_dict(self):
        with open(self.target_file) as reader:
            texts = reader.readlines()
            self.wave_text_dict = {}
            for i in texts:
                id, text = i.strip().split('\t')
                assert id not in self.wave_text_dict
                self.wave_text_dict[id] = text
        print(f'total {len(self.wave_text_dict)} wave text pairs.')

    def build_wave_list(self):
        self.wave_list = []
        self.wave_id_dict = {}
        sub_folder_count = 0
        wave_file_count = 0
        for sub_folder in os.listdir(self.wave_folder):
            sub_folder_count += 1
            for file in os.listdir(os.path.join(self.wave_folder, sub_folder)):
                wave_file = os.path.join(self.wave_folder, sub_folder, file)
                self.wave_id_dict[wave_file] = file.split('.')[0]
                self.wave_list.append(wave_file)
                wave_file_count += 1
        print(f'total {sub_folder_count} sub_folder, {wave_file_count} wave_file.')

    def get_target(self, filepath):
        id = self.wave_id_dict[filepath]
        text = self.wave_text_dict[id]
        return text

    def __getitem__(self, item):
        wave_file_path = self.wave_list[item]
        return {'wave': wave_file_path, 'tgt':self.get_target(wave_file_path)}

    def __len__(self):
        return len(self.wave_list)


@dataclass
class RawCollector_1:
    pass




# raw_collector = RawCollector()
# for i in raw_collector:
#     print(i)
#     break


@dataclass
class SampleFactory:
    wave_file_path: str
    text: str
    wave_feature: t.Tensor = None
    text_id: list = None

    @classmethod
    def build_sample(cls, wave_file_path, text, vocab, **kwargs):
        obj = cls(wave_file_path, text)
        obj.build_text(vocab)
        obj.build_feature(kwargs)
        return obj

    def build_text(self, vocab):
        self.text_id = vocab.numericalize(self.text, True, True)

    def build_feature(self, **kwargs):
        tensor, sample_rate = ta.load(self.wave_file_path, normalization=True)
        self.sample_rate = sample_rate
        transform = ta.transforms.MFCC(sr=16000, n_mfcc=40, log_mels=True, melkwargs={'ws': 25, 'hop': 20})
        self.wave_feature = transform(tensor)


@dataclass
class SampleStorge:
    samples = []
    samples_dev = []

    def __repr__(self):
        return f'storge with {len(self.samples)} samples'

    def add_sample(self, sample):
        self.samples.append(sample)

    def save(self, path):
        all = [self.samples, self.samples_dev]
        t.save(all, path)

    def load(self, path):
        all = t.load(path)
        self.samples = all[0]
        self.samples_dev = all[1]

    def split(self, test_size):
        assert self.samples_dev == []
        self.samples, self.samples_dev = train_test_split(self.samples, test_size=test_size)

    def filter(self, wave_max, wave_min):
        ori = len(self.samples)
        self.samples = [i for i in self.samples if len(i.wave_feature[0]) > wave_min and len(i.wave_feature[0]) < wave_max]
        now = len(self.samples)
        print(f'filtered {ori - now} samples')


class Padder:

    @staticmethod
    def pad_two(inputs, pad_value, lengths=None):
        if lengths is None:
            lengths = [len(i) for i in inputs]
        batch_size = len(inputs)
        output = t.full((batch_size, max(lengths)), pad_value)
        for index, (i, l) in enumerate(zip(inputs, lengths)):
            if isinstance(i, list):
                n = t.Tensor(i)
            else:
                n = i
            output[index, :l] = n
        return output, lengths

    @staticmethod
    def pad_tri(inputs, pad_value, lengths=None):
        if lengths is None:
            lengths = [len(i) for i in inputs]
        batch_size = len(inputs)
        output = t.full((batch_size, max(lengths), inputs[0].size(-1)), pad_value)
        for index, (i, l) in enumerate(zip(inputs, lengths)):
            output[index, :l, :] = i
        return output, lengths