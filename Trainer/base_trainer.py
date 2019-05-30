import torch as t
from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np
from typing import Any
from collections import defaultdict
import shutil
from tqdm import tqdm
from dataclasses import dataclass


class MetricsManager(object):
    """
    MetricsManager
    """
    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics):
        """
        update
        """
        num_samples = metrics.pop("num_samples", 1)
        self.num_samples += num_samples

        for key, val in metrics.items():
            if val is not None:
                if isinstance(val, t.Tensor):
                    val = val.item()
                    self.metrics_cum[key] += val * num_samples
                else:
                    assert len(val) == 2
                    val, num_words = val[0].item(), val[1]
                    self.metrics_cum[key] += np.array(
                        [val * num_samples, num_words])
                self.metrics_val[key] = val

    def clear(self):
        """
        clear
        """
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def get(self, name):
        """
        get
        """
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        """
        report_val
        """
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{}-{:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        """
        report_cum
        """
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{}-{:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)

            if num_words is not None:
                ppl = np.exp(min(val / num_words, 100))
                metric_str = "{}_PPL-{:.3f}".format(key.upper(), ppl)
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs


@dataclass
class BaseTrainer:
    optimizer: Any
    model: Any
    train_iter: Any
    dev_iter: Any
    test_iter: Any
    ckpt_root: str = 'ckpt/'
    exp_name: str = 'base_exp/'
    eval_every_iter: int = 1000
    save_every_iter: int = 5000
    drop_exp: bool = True

    def __post_init__(self):
        self.exp_root = os.path.join(self.ckpt_root, self.exp_name)
        print(os.listdir(self.ckpt_root))
        self.global_step = 0
        self.global_epoch = 0
        if self.drop_exp:        #TODO delete if use
            shutil.rmtree(self.exp_root)
        # assert not os.path.exists(self.exp_root)
        os.mkdir(self.exp_root)
        self.summary_writer = SummaryWriter(self.exp_root)
        self.config = self.model.config

    def train(self, from_ckpt=None):
        if from_ckpt is not None:
            self.load_from_ckpt(from_ckpt)
        for i in range(self.config.num_epoch):
            self.train_epoch()
            self.global_epoch += 1

    def train_epoch(self):
        self.model.train()
        for data in tqdm(self.train_iter):
            loss, score = self.model.iterate(data, optimizer=self.optimizer, is_train=True)
            self.global_step += 1
            if self.global_step % self.eval_every_iter == 0 and self.global_step != 0:
                self.evaluate(self.dev_iter)

            if self.global_step % self.save_every_iter == 0 and self.global_step != 0:
                self.save_ckpt()
        self.save_ckpt()
        self.evaluate(self.test_iter)

    def load_from_ckpt(self, from_ckpt):
        model_file = os.path.join(from_ckpt, )
        opt_file = os.path.join(from_ckpt, )
        self.model.load(model_file)
        self.optimizer.load(opt_file)
        self.config = self.model.config
        print(f'train state loaded from {from_ckpt}')

    def save_ckpt(self, ckpt):
        model_file = os.path.join(ckpt, )
        opt_file = os.path.join(ckpt, )
        self.model.save(model_file)
        self.optimizer.save(opt_file)
        print(f'train state saved to {ckpt}')

    def copy_best(self):
        pass

    def evaluate(self, dev_iter):
        print(f'\n evaluating')
        self.model.eval()
        for data in tqdm(dev_iter):
            output = self.model.iterate(data, is_train=False)
        self.model.train()

    def get_time(self):
        return time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))

