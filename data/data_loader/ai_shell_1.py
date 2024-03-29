import json
import torch as t
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from Predictor.data_handler import AudioParser, AudioParser2
from Predictor.data_handler import Padder
from Predictor.Utils import Pack


class AiShell1(Dataset):
    """
    parser paras, some is not implemented
        sample_rate: int = 16000
        n_mels: int = 40
        window_size: int = 400
        hop: int = None
        f_min: int = 0
        f_max: int = None
        pad: int = 0
    """
    def __init__(self, datas, vocab, sample_rate=16000, window_size=400, n_mels=40, augment=False, use_old=False):
        super(AiShell1, self).__init__()
        self.datas = datas
        self.vocab = vocab

        self.parser = AudioParser(sample_rate=sample_rate, n_mels=n_mels, window_size=window_size)
        self.augment = augment
        self.use_old = use_old

    def filter(self, lenths):
        n = []
        l = len(self.datas)
        for i, v in tqdm(enumerate(self.datas), desc=f'filter train data longer than {lenths}.'):
            wave, _, _, _ = self.__getitem__(i)
            l1 = wave.shape[0]
            if l1 < lenths:
                n.append(v)
        self.datas = n

        print(f'filter done, {l - len(self.datas)} sample filtered, {len(self.datas)} sample left.')

    def __getitem__(self, item):
        line = json.loads(self.datas[item])

        if self.use_old:
            file = line['wave'].split('.')[0] + '.t'
            wave, tgt_for_input, tgt_for_metric = t.load(file)
            return wave, tgt_for_input, tgt_for_metric, line
        else:
            tgt_for_input = self.vocab.convert_str(line['tgt'], use_bos=False, use_eos=False)
            tgt_for_metric = self.vocab.convert_str(line['tgt'], use_bos=False, use_eos=False)
            wave = self.parser.parse(line['wave'], augment=self.augment)
            return wave, tgt_for_input, tgt_for_metric, line

    def __len__(self):
        return len(self.datas)

    def pre_dump_features(self):
        for i in tqdm(self, desc='pre dumping features'):
            wave, tgt_for_input, tgt_for_metric, line = i
            file = line['wave'].split('.')[0] + '.t'
            t.save((wave, tgt_for_input, tgt_for_metric), file)


@dataclass
class collat:
    """
    collate function to replace fn collate_fn
    for use cuda choice
    """
    use_cuda: bool = True

    def __call__(self, batch):
        pack = Pack()
        wave = [i[0] for i in batch]
        tgt_for_input = [i[1] for i in batch]
        tgt_for_metric = [i[2] for i in batch]
        wave, wave_len = Padder.pad_tri(wave, 0)
        tgt_for_input, tgt_for_input_len = Padder.pad_two(tgt_for_input, 0)
        tgt_for_metric, tgt_for_metric_len = Padder.pad_two(tgt_for_metric, 0)
        pack.add(wave=wave, tgt_for_input=tgt_for_input.long(), wave_len=t.Tensor(wave_len).long(), tgt_len=t.Tensor(tgt_for_input_len).long())
        pack.add(tgt_for_metric=tgt_for_metric.long())
        if self.use_cuda:
            pack = pack.cuda()

        return pack


def build_dataloader(collector_path, vocab, batch_size, part='test', use_cuda=True,
                     sample_rate=16000, window_size=400, n_mels=40, augment=False, predump=True, use_old=True):
    with open(collector_path + '_' + part + '.json') as reader:
        datas = reader.readlines()
    data_set = AiShell1(datas, vocab, sample_rate=sample_rate, window_size=window_size, n_mels=n_mels, augment=augment,
                        use_old=use_old)
    # if part == 'train':
    #     data_set.filter(800)
    if predump:
        data_set.use_old = False
        data_set.pre_dump_features()
        data_set.use_old = True
    data_loader = DataLoader(data_set, batch_size, collate_fn=collat(use_cuda), drop_last=True)
    return data_loader

if __name__ == '__main__':
    pass
