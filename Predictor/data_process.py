import torchaudio as ta
from dataclasses import dataclass
import torch as t
import os


@dataclass
class RawCollector:
    wave_folder = 'data/wav/'
    target_file = 'data/trans.txt'

    def __post_init__(self):
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
        return wave_file_path, self.get_target(wave_file_path)

# raw_collector = RawCollector()
# for i in raw_collector:
#     print(i)
#     break


@dataclass
class SampleFactory:
    wave_file_path: str
    text: str
    text_id: list = None

    @classmethod
    def get_sample(cls, wave_file_path, text, vocab):
        return cls(wave_file_path, text).build_feature().build_text(vocab)

    def build_text(self, vocab):
        tensor, sample_rate = ta.load(self.wave_file_path, normalization=True)


    def build_feature(self):
        raise NotImplementedError

# raw_collector = RawCollector()
# for i in raw_collector:
#     print(i)
#     break
# sample = BaseSampleFactory.get_sample(*i)


class MelSampleFactory:

    def __init__(self):
        super(MelSampleFactory, self).__init__()

    def build_feature(self):
        wave = ta.load(self.wave_file_path, normalization=True)
        pass

    def build_text(self):
        pass

