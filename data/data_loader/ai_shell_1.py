import torch as t
from torch.utils.data import Dataset, DataLoader
from Predictor.data_handler.vocab import Vocab
from Predictor.data_handler.processor import AudioParser
from Predictor.data_handler.padder import Padder
from Predictor.Utils import Pack
import json
from dataclasses import dataclass


padder = Padder()


class AiShell1(Dataset):
    def __init__(self, datas, vocab, argument=False):
        super(AiShell1, self).__init__()
        self.datas = datas
        self.vocab = vocab
        self.parser = AudioParser()

    def __getitem__(self, item):
        line = json.loads(self.datas[item])
        tgt = self.vocab.convert_str(line['tgt'])
        wave = self.parser.parse(line['wave'])
        return wave, tgt

    def __len__(self):
        return len(self.datas)

@dataclass
class collat:
    """
    collate function to replace fn collate_fn
    for use cuda choice
    """
    use_cuda: bool = True

    def __call__(self, batch):
        pack = Pack()
        waves = [i[0] for i in batch]
        tgt = [i[1] for i in batch]
        waves, wave_len = Padder.pad_tri(waves, 0)
        tgt, tgt_len = Padder.pad_two(tgt, 0)
        pack.add(waves=waves, tgt=tgt.long(), wave_len=wave_len, tgt_len=tgt_len)
        return pack


def build_dataloader(collector_path, vocab_path, batch_size, part='test', use_cuda=True):
    with open(collector_path + '_' + part + '.json') as reader:
        datas = reader.readlines()
    vocab = Vocab.load(vocab_path)
    data_set = AiShell1(datas, vocab)
    data_loader = DataLoader(data_set, batch_size, collate_fn=collat(use_cuda))
    return data_loader, vocab

