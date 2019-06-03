import fire
import torch as t

from Predictor import Models
from Predictor.data_handler.data_config import DataConfigAiShell1
from data.data_loader.ai_shell_1 import build_dataloader
from Trainer.base_trainer import BaseTrainer
from Trainer.optimizer import NoamOpt


class TrainConfig(DataConfigAiShell1):
    lr = 1e-3
    batch_size = 16
    eval_batch_size = 32
    num_epoch = 10
    device_id = (0, 1)
    exp_name = 'test3'
    drop_exp = True
    ckpt_root: str = 'ckpt/'
    log_every_iter: int = 100
    eval_every_iter: int = 20000
    save_every_iter: int = 5000
    reference = '-loss'
    from_ckpt = None


def get_model_class(model_name):
    Model = getattr(Models, model_name)
    ModelConfig = Model.get_default_config()
    return Model, ModelConfig


def train(**kwargs):
    print('\nStart training\n')
    assert kwargs['model_name']
    Model, ModelConfig = get_model_class(kwargs['model_name'])
    model_config = ModelConfig()
    config = TrainConfig()
    config.fn_combine(model_config)
    config.fn_build(kwargs)
    config.fn_show()

    train_iter, vocab = build_dataloader(
        collector_path=config.collector_path, vocab_path=config.vocab_path, batch_size=config.batch_size, part='train')
    test_iter, _ = build_dataloader(
        collector_path=config.collector_path, vocab_path=config.vocab_path, batch_size=config.eval_batch_size, part='test')
    dev_iter, _ = build_dataloader(
        collector_path=config.collector_path, vocab_path=config.vocab_path, batch_size=config.eval_batch_size, part='dev')

    model = Model(config, vocab)
    optimizer = t.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = NoamOpt(config.hidden_size, 1, 4000, optimizer)
    trainer = BaseTrainer(
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

    fire.Fire(train, '--lr=1 --model_name="ExampleModel" --batch_size=32 --exp_name="test4" --drop_exp=False')
