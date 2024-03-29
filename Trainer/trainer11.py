import os
import shutil
import torch
import datetime
from typing import Any
from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

from Trainer import MetricsManager


@dataclass
class Trainer11:
    optimizer: Any
    model: Any
    train_iter: Any
    dev_iter: Any
    test_iter: Any
    ckpt_root: str = 'ckpt/'
    exp_name: str = 'base_exp'
    log_every_iter: int = 100
    eval_every_iter: int = 1000
    save_every_iter: int = 5000
    drop_exp: bool = True
    reference = '-loss'

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = self.get_time()
        self.exp_root = os.path.join(self.ckpt_root, self.exp_name)
        self.global_step = 0
        self.global_epoch = 0
        if self.drop_exp and os.path.exists(self.exp_root):        #TODO delete if use
            print(f'droped {self.exp_root}')
            shutil.rmtree(self.exp_root)
        os.mkdir(self.exp_root)
        self.summary_writer = SummaryWriter(self.exp_root)
        self.config = self.model.config
        assert self.reference[0] in ['-', '+']

    def train(self, from_ckpt=None, from_epoch=None, from_step=None):
        self.best = 1e10 if self.reference[0] == '-' else 0
        if from_ckpt is not None and from_epoch is not None and from_step is not None:
            self.load_from_ckpt(exp_name=from_ckpt, epoch=from_epoch, step=from_step)

        for i in range(self.config.num_epoch):
            self.train_epoch()
            self.global_epoch += 1

    def train_epoch(self):
        self.model.train()
        max_len = 0
        train_bar = tqdm(iterable=self.train_iter, leave=True, total=len(self.train_iter))
        average_loss = 0
        for i, data in enumerate(train_bar):
            metrics, _ = self.model.iterate(data, optimizer=self.optimizer, is_train=True)
            lr = self.optimizer.rate()
            self.summary_writer.add_scalar('lr', lr, self.global_step)

            if self.global_step % self.log_every_iter == 0 and self.global_step != 0:
                self.summarize(metrics, 'train/')

            self.global_step += 1
            if self.global_step % self.eval_every_iter == 0 and self.global_step != 0:
                self.evaluate(self.dev_iter, 'dev/')

            if self.global_step % self.save_every_iter == 0 and self.global_step != 0:
                self.save_ckpt()
            le = data.wave.size(1)
            if le > max_len:
                max_len = le
            average_loss += metrics.loss.item()
            desc = f'epoch: {self.global_epoch}, lr:{round(self.optimizer._rate, 6)}, max_len: {max_len}, loss: {round(average_loss / (i+1), 4)}, current loss:{round(metrics.loss.item(), 4)} cer: {round(metrics.cer.item(), 4)}'
            train_bar.set_description(desc)
        if self.global_epoch in [10, 50, 80, 100, 200]:
            self.evaluate(self.dev_iter, 'dev/')
        #print(f'in train epoch:{self.global_epoch}, average_loss{1} average_score{1}')#TODO use true value
        self.save_ckpt()
        self.evaluate(self.test_iter, 'test/')

    def load_from_ckpt(self, exp_name, epoch, step):
        prefix = f'e{epoch}_s{step}'
        self.global_step = step
        self.global_epoch = epoch
        model_file = os.path.join(self.ckpt_root, exp_name, prefix+'.model')
        opt_file = os.path.join(self.ckpt_root, exp_name, prefix+'.opt')
        self.model.load(model_file)
        self.optimizer.load(opt_file)
        self.config = self.model.config
        print(f'train state loaded from {os.path.join(self.ckpt_root, exp_name)}_epoch:{epoch} step:{step}\n')

    def save_ckpt(self):
        prefix = f'e{self.global_epoch}_s{self.global_step}'
        model_file = os.path.join(self.exp_root, prefix+'.model')
        opt_file = os.path.join(self.exp_root, prefix+'.opt')
        self.model.save(model_file)
        self.optimizer.save(opt_file)
        print(f'train state saved to {self.exp_root}_epoch:{self.global_epoch} step:{self.global_step}\n')
    #     if (self.reference[0] == '-' and reference_score < self.best) or\
    #             (self.reference[0] == '+' and reference_score > self.best):
    #         self.copy_best()
    #         self.best = reference_score
    #
    # def copy_best(self):
    #     shutil

    def summarize(self, pack, prefix='train/'):
        # print(f'\nsummarizing in {self.global_step}')
        for i in pack:
            tmp_prefix = prefix + i
            self.summary_writer.add_scalar(tmp_prefix, pack[i].detach().cpu().numpy(), self.global_step)

    def evaluate(self, dev_iter, prefix='dev/'):
        print(f'\nEvaluating')
        self.model.eval()
        dev_metric_manager = MetricsManager()
        dev_bar = tqdm(dev_iter, leave=True, total=len(dev_iter), disable=False)
        with torch.no_grad():
            for data in dev_iter:
                metrics, _ = self.model.iterate(data, is_train=False)
                dev_metric_manager.update(metrics)
                desc = f'Valid-loss: {metrics.loss.item()}, cer: {1}'
                dev_bar.set_description(desc)
            print(f'\nValid, average_loss: {1}, average_score: {1}')#TODO use true value
            report = dev_metric_manager.report_cum()
            report = dev_metric_manager.extract(report)
            self.summarize(report, 'dev/')
            self.model.train()

    def get_time(self):
        return (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d%H%M")

