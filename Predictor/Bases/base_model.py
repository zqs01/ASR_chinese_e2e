import numpy as np
from torch.nn.parallel import DataParallel
import torch as t
import os
from dataclasses import dataclass
from Predictor.Bases.base_config import BaseConfig


class Wrapper(t.nn.Module):
    def __init__(self, model, device_ids: list):
        super(Wrapper, self).__init__()
        self.model = DataParallel(model.cuda(), device_ids)

    def forward(self, *input):
        return self.model.forward(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)


class BaseModel(t.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def forward(self, *input):
        raise NotImplementedError

    def iterate(self, *inputs):
        raise NotImplementedError

    def cal_metrics(self, *inputs):
        raise NotImplementedError

    def num_para(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return f'Trainable parameters:{params}'

    def save(self, path):
        t.save(self.state_dict(), path)
        print(f'\nmodel saved to {path}')

    def load(self, path):
        if os.path.isfile(path):
            state_dict = t.load(
                path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("\nLoaded model state from '{}'".format(path))
        else:
            print("\nInvalid model state file: '{}'".format(path))

    @classmethod
    def get_default_config(cls):
        @dataclass
        class ModelConfig(BaseConfig):
            pass
            # emb = 100

        return ModelConfig

    def wrap(self, device_ids: list = (0, 1)):
        assert t.cuda.is_available()
        print(f'wrapped model to GPU:{device_ids}')
        return Wrapper(self, device_ids)

