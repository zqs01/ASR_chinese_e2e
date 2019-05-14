import torch as t
from Predictor.Utils.data_processor import SampleStorge
from torch.utils.data import DataLoader, Dataset


class AudioSet(Dataset):
    def __init__(self, storge_path, ):
        self.storge = SampleStorge.load(storge_path)

    def build(self):
        pass

    def __getitem__(self, item):
        pass
