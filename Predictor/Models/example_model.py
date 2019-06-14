import torch as t
from Predictor.Bases import BaseConfig
from Predictor.Bases import BaseModel
from Predictor.Utils import Pack
from dataclasses import dataclass
import os


class ExampleModel(BaseModel):
    def __init__(self, config, vocab):
        super(ExampleModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.linear = t.nn.Linear(3, 4)

    def forward(self, pack):
        output = Pack()
        wave, wave_length = pack.wave, pack.wave_len
        output.add(wave=wave)
        return output

    def cal_metric(self, model_output):
        metrics = Pack()
        metrics.add(loss=model_output.wave.sum())
        return metrics, None

    def iterate(self, input, optimizer=None, is_train=True):
        output = self.forward(input)
        metrics, scores = self.cal_metric(output)

        # loss = metrics.loss
        loss = t.Tensor([0])
        loss.requires_grad = True
        if t.isnan(loss):
            raise ValueError("nan loss encountered")
        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=5.0)
            optimizer.step()

        return metrics, None

    @classmethod
    def get_default_config(cls):
        @dataclass
        class ModelConfig(BaseConfig):
            emb = 100
            hidden_size = 300

        return ModelConfig

    def save(self, path):
        all = {'state_dict': self.state_dict(), 'config': self.config, 'vocab': self.vocab}
        t.save(all, path)
        print(f'\nmodel saved to {path}')

    def load(self, path):
        if os.path.isfile(path):
            all = t.load(path)
            self.load_state_dict(all['state_dict'], strict=False)
            self.config = all['config']
            self.vocab = all['vocab']
            print("\nLoaded model state from '{}'".format(path))
        else:
            print("\nInvalid model state file: '{}'".format(path))