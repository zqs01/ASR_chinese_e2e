import fire
from Predictor.Models.transformer import ModelConfig, Model
from Predictor.Utils.vocab import Vocab
from preprocess import VOCAB_FILE, STORGE_FILE
from trainner import Trainner
from dataloader import AudioSet


CONFIG_FILE = ''

class ExpConfig(ModelConfig):
    # experiments configs
    vocab_file = VOCAB_FILE
    storge_file = STORGE_FILE
    config_file = CONFIG_FILE

    lr = 1e-3
    batch_size = 32
    num_epoch = 20
    eval_every_step = 100
    exp_root = 'ckpt/'



def train(**kwargs):
    config = ExpConfig.build(kwargs)
    vocab = Vocab(load_from=config.vocab_file)
    model = Model(vocab, config)
    train_loader = AudioSet.build_loader('train', batch_size=config.batch_size)
    dev_loader = AudioSet.build_loader('dev', batch_size=config.batch_size)


    trainner = Trainner()
    trainner.train()

def eval():
    pass


if __name__ == '__main__':
    fire.Fire()