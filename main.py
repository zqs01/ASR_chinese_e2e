import fire
import torch as t

from Predictor import Models
from Predictor.data_handler import Vocab
from Predictor.data_handler import DataConfigAiShell1
from data.data_loader.ai_shell_1 import build_dataloader
from Trainer import Trainer11
from Trainer import NoamOpt



class TrainConfig(DataConfigAiShell1):
    lr = 1e-3
    batch_size = 16
    eval_batch_size = 16
    num_epoch = 1000
    warm_up = 4000
    device_id = (0, 1)
    exp_name = None
    drop_exp = True
    ckpt_root: str = 'ckpt/'
    log_every_iter: int = 100
    eval_every_iter: int = 20000
    save_every_iter: int = 5000
    reference = '-loss'
    from_ckpt = None # TODO not implemented !
    model_name = 'ExampleModel'
    predump = True ## pre dump feature when build data sets
    use_old = True ## use dumped feature when data loader


def get_model_class(model_name):
    Model = getattr(Models, model_name)
    ModelConfig = Model.get_default_config()
    return Model, ModelConfig


def show_configs(**kwargs):
    print('\nStart training\n')
    config = TrainConfig()
    assert config.model_name
    Model, ModelConfig = get_model_class(config.model_name)
    model_config = ModelConfig()
    config.fn_combine(model_config)
    config.fn_build(kwargs)
    config.fn_show()


def train(**kwargs):
    print('\nStart training\n')
    config = TrainConfig()
    config.fn_build(kwargs)
    assert config.model_name
    Model, ModelConfig = get_model_class(config.model_name)
    model_config = ModelConfig()
    config.fn_combine(model_config)
    config.fn_build(kwargs)
    config.fn_show()
    vocab = Vocab.load(config.vocab_path)
    train_iter = build_dataloader(
        collector_path=config.collector_path, vocab=vocab, batch_size=config.batch_size, part='train',
        sample_rate=config.sample_rate, window_size=config.window_size, n_mels=config.n_mels, augment=config.augment,
        predump=config.predump, use_old=config.use_old)
    test_iter = build_dataloader(
        collector_path=config.collector_path, vocab=vocab, batch_size=config.eval_batch_size, part='test',
        sample_rate=config.sample_rate, window_size=config.window_size, n_mels=config.n_mels, augment=False,
        predump=config.predump, use_old=config.use_old)
    dev_iter = build_dataloader(
        collector_path=config.collector_path, vocab=vocab, batch_size=config.eval_batch_size, part='dev',
        sample_rate=config.sample_rate, window_size=config.window_size, n_mels=config.n_mels, augment=False,
        predump=config.predump, use_old=config.use_old)

    model = Model(config, vocab)
    model = model.wrap()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    #assert config.hidden_size
    #optimizer = NoamOpt(config.hidden_size, 1, config.warm_up, optimizer)
    trainer = Trainer11(
        model=model,
        optimizer=optimizer,
        train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter,
        exp_name=config.exp_name,
        eval_every_iter=config.eval_every_iter,
        log_every_iter=config.log_every_iter,
        save_every_iter=config.save_every_iter
    )
    print(f'start trainning at {trainer.get_time()}\n')
    trainer.train()
    print(f'done at {trainer.get_time()}\n')


if __name__ == '__main__':
    #fire.Fire(show_configs)
    fire.Fire(train, '--lr=1e-3 --model_name="Transformer" --batch_size=16 --drop_exp=False --predump=False --warm_up=200 --log_every_step=1')
