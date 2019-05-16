import torch as t
from Predictor.Utils.data_processor import SampleStorge
from torch.utils.data import DataLoader, Dataset
import numpy as np
from preprocess import STORGE_FILE
from Predictor.Utils.data_processor import Padder


class AudioDataset(Dataset):
    def __init__(self, storge_file=STORGE_FILE, part='train'):
        super(AudioDataset, self).__init__()
        self.storge = SampleStorge()
        self.storge.load(storge_file)
        self.storge.filter(700, 100)
        self.part = part

    def __getitem__(self, item):
        if self.part == 'train':
            data = self.storge.samples[item]
            return data.wave_feature.squeeze(0), data.text_id

    def __len__(self):
        if self.part == 'train':
            return len(self.storge.samples)
        elif self.part == 'dev':
            return len(self.storge.samples_dev)

    def create_dataloader(self, batch_size, collate_fn=collate_fn):
        loader = DataLoader(self, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
        return loader


def collate_fn(batch):
    wave_feature = [i[0] for i in batch]
    target = [i[1] for i in batch]
    wave_feature, wave_length = Padder.pad_tri(wave_feature, 0)
    target, target_length = Padder.pad_two(target, 0)
    return wave_feature, wave_length, target, target_length


# dataset = AudioDataset()
# loader = DataLoader(dataset, 2, collate_fn=collate_fn)

