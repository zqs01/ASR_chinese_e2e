import torch as t
from tensorboardX import SummaryWriter



class Trainner:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

