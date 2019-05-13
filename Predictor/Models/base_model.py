import torch as t
from Predictor.config_factory import ConfigFactory


class ModelConfig(ConfigFactory):
    hidden_size = 64
    model_name = 'base_model'


class BaseModel(t.nn.Module):
    def __init__(self, vocab, config):
        super(BaseModel, self).__init__()


    def forward(self, *input):
        raise NotImplementedError

    def save(self, path):
        #TODO
        pass

    def load(self, path):
        #TODO
        pass


