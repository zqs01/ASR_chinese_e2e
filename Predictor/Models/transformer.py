import torch as t
from Predictor.Models.base_model import BaseModel
from Predictor.config_factory import ConfigFactory


class ModelConfig(ConfigFactory):
    hidden_size = 64
    num_head = 8
    model_name = 'transformer'

class Model(BaseModel):
    #TODO build real model

    def __init__(self, vocab, config):
        super(Model, self).__init__(vocab, config)
        pass



