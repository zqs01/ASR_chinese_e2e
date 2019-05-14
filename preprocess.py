from Predictor.Utils.data_processor import SampleStorge, SampleFactory, RawCollector
from Predictor.Utils.vocab import Vocab
import fire
from tqdm import tqdm


VOCAB_FILE = 'Predictor/vocab.t'
STORGE_FILE = 'data/storge.t'


def preprocess():
    collector = RawCollector()
    vocab = Vocab()
    vocab.build(collector, build_fields=['tgt'])
    vocab.dump_vocab(VOCAB_FILE)
    storge = SampleStorge()

    for line in tqdm(collector):
        sample = SampleFactory.build_sample(line['wave'], line['tgt'], vocab)
        storge.add_sample(sample)
    print(storge)
    storge.save(STORGE_FILE)


if __name__ == '__main__':
    fire.Fire()

