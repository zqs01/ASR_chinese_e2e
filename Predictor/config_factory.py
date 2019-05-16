import torch as t
from dataclasses import dataclass

@dataclass
class ConfigFactory:

    @classmethod
    def build(cls, kwargs):
        obj = cls()
        for k, v in kwargs.items():
            print(k, v)
            if hasattr(obj, k):
                if getattr(obj, k) != v:
                    print(f'changed {k}:{getattr(obj, k)} to {v}')
                    setattr(obj, k, v)
            else:
                setattr(obj, k, v)
                print(f'add {k}:{v}')
        return obj

    def save(self, path):
        t.save(self, path)

    def load(self, path):
        self = t.load(path)
        return self

    @property
    def contents(self):
        for k, v in self.__dict__.items():
            print(f'{k}: {v}')

    def combine(self, config):
        for k, v in config.__dict__.items():
            print(k, v)
            if hasattr(self, k):
                if getattr(self, k) != v:
                    print(f'changed {k}:{getattr(self, k)} to {v}')
                    setattr(self, k, v)
            else:
                setattr(self, k, v)
                print(f'add {k}:{v}')


