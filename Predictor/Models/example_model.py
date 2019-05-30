import torch as t
from Predictor.Bases.base_config import BaseConfig
from Predictor.Utils.pack import Pack
from dataclasses import dataclass


class ExampleModel(t.nn.Module):
    def __init__(self, config, vocab):
        super(ExampleModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.linear = t.nn.Linear(3, 4)

    def cal_metric(self, model_output):
        metrics = Pack()

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

        return output, scores

    def forward(self, pack):
        wave, wave_length = pack.wave, pack.wave_len

        return pack

    @classmethod
    def get_default_config(cls):
        @dataclass
        class ModelConfig(BaseConfig):
            emb = 100

        return ModelConfig

