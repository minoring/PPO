import os
import time

from torch.utils.tensorboard import SummaryWriter

import pytorch_utils as ptu


class Logger:
    def __init__(self, env_name, log_dir, split='train'):
        self.env_name = env_name
        self.split = split
        self.log_dir = os.path.join(log_dir, env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.ep_rewards = []
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, scalar, step):
        self.writer.add_scalar(f'{name}/{self.split}', scalar, step)
