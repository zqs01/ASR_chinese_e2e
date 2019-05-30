import torch as t
from torch.nn import MultiheadAttention
from torch.nn import LayerNorm
from Predictor.Bases.base_model import BaseModel, BaseConfig
from dataclasses import dataclass


class Transformer(BaseModel):
    def __init__(self, config, vocab):
        super(Transformer, self).__init__()
        self.config = config
        self.vocab = vocab
        self.conv = Conv()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = Output()

    def forward(self, *input):
        pass

    def cal_metrics(self, *inputs):
        pass

    def iterate(self, *inputs):
        pass

    @classmethod
    def get_default_config(cls):
        @dataclass
        class ModelConfig(BaseConfig):
            num_layer = 6
            feed_forward_size = 256
            hidden_size = 128

        return ModelConfig


    def greedy_search(self):
        pass

    def beam_search(self):
        pass





