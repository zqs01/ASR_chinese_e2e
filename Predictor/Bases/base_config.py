import torch as t
from dataclasses import dataclass

@dataclass
class BaseConfig:

    def fn_build(self, kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if getattr(self, k) != v:
                    print(f'\tchanged {k}:{getattr(self, k)} to {v}')
                    setattr(self, k, v)
            else:
                setattr(self, k, v)
                print(f'\tadd {k}:{v}')

    def fn_save(self, path):
        t.save(self, path)

    def fn_load(self, path):
        self = t.load(path)
        return self

    def fn_show(self):

        print('\nconfigs: ')
        for att in self.__dir__():
            if not att.startswith('_') and not att.startswith('__') and not att.startswith('fn_'):
                print(f'\t{att}:\t\t{getattr(self, att)}')
        print('\n')

    def fn_get_attrs(self):
        for att in self.__dir__():
            if not att.startswith('_') and not att.startswith('__') and not att.startswith('fn_'):
                yield att, getattr(self, att)

    def fn_combine(self, config):
        print(f'combining configs')
        for k, v in config.fn_get_attrs():
            if hasattr(self, k):
                if getattr(self, k) != v:
                    print(f'\tchanged {k}:{getattr(self, k)} to {v}')
                    setattr(self, k, v)
            else:
                setattr(self, k, v)
                print(f'\tadd {k}:{v}')
