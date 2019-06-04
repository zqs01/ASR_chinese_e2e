import os
import json
from tqdm import tqdm
from dataclasses import dataclass

from Predictor.data_handler import Vocab


@dataclass
class CollectorAiShell1:
    """
    collector for aishell 1
    """
    data_folder: str = 'data/data_aishell/'

    def __post_init__(self):
        self.wave_folder: str = self.data_folder + 'wav/'
        self.target_file: str = self.data_folder + 'transcript/aishell_transcript_v0.8.txt'
        self.datas = {}
        self.datasets = os.listdir(self.wave_folder)
        self.drop_count = 0
        self.load_wave_text_dict()
        for dataset in self.datasets:
            self.build_wave_list(dataset)

    def load_wave_text_dict(self):
        with open(self.target_file) as reader:
            texts = reader.readlines()
            self.wave_text_dict = {}
            for i in texts:
                line = i.strip().split(' ')
                id = line[0]
                text = ''.join(line[1:])
                assert id not in self.wave_text_dict
                self.wave_text_dict[id] = text
        print(f'total {len(self.wave_text_dict)} wave text pairs.')

    def build_wave_list(self, data_set='train'):
        wave_list = []
        wave_id_dict = {}
        sub_folder_count = 0
        wave_file_count = 0
        for sub_folder in os.listdir(os.path.join(self.wave_folder, data_set)):
            sub_folder_count += 1
            for file in os.listdir(os.path.join(self.wave_folder, data_set, sub_folder)):
                if file.split('.')[0] in self.wave_text_dict:
                    wave_file = os.path.join(self.wave_folder, data_set, sub_folder, file)
                    wave_id_dict[wave_file] = file.split('.')[0]
                    wave_list.append(wave_file)
                    wave_file_count += 1
                else:
                    self.drop_count += 1
        print(f'{data_set}:total {sub_folder_count} sub_folder, {wave_file_count} wave_file. {self.drop_count} droped')
        self.datas[data_set] = {'wave_list': wave_list, 'wave_id_dict': wave_id_dict}

    def get_target(self, file_path, data_set):
        id = self.datas[data_set]['wave_id_dict'][file_path]
        text = self.wave_text_dict[id]
        return text

    def items(self, data_set):
        assert data_set in self.datasets
        for wave_file_path in self.datas[data_set]['wave_list']:
            yield {'wave': wave_file_path, 'tgt': self.get_target(wave_file_path, data_set)}

    def build_vocab(self, path='Predictor/vocab.t'):
        self.vocab = Vocab()
        for i in tqdm(self.items('train')):
            self.vocab.consume_sentance(i['tgt'])
        self.vocab.build()
        self.vocab.save(path)

    def save(self, path):
        for dataset in self.datasets:
            with open(path+'_'+dataset+'.json', 'w') as writer:
                for line in self.items(dataset):
                    json.dump(line, writer, ensure_ascii=False)
                    writer.write('\n')
        print(f'collector saved in {path}')


if __name__ == '__main__':
    collecotr = CollectorAiShell1(data_folder='../../data/data_aishell/')
    collecotr.build_vocab(path='../../Predictor/vocab.t')
    collecotr.save('data/data_collector/collector')
