import os
import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, env_name, log_dir, log_csv_path, seed, split='train'):
        self.env_name = env_name
        self.log_dir = os.path.join(log_dir, env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.ep_rewards = []
        self.seed = seed
        self.split = split
        self.log_csv_path = log_csv_path
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, scalar, step):
        self.writer.add_scalar(f'{name}/{self.split}', scalar, step)

    def save(self):
        env_names = [self.env_name] * len(self.ep_rewards)
        splits = [self.split] * len(self.ep_rewards)
        ep_idxs = list(range(1, len(self.ep_rewards) + 1))
        seeds = [self.seed] * len(self.ep_rewards)

        df = pd.DataFrame(list(zip(env_names, splits, ep_idxs, self.ep_rewards, seeds)),
                          columns=['Env', 'Split', 'Episode', 'Reward', 'Seed'])

        df.to_csv(self.log_csv_path,
                  mode='a',
                  header=not os.path.exists(self.log_csv_path),
                  index=False)
        print(f'Log saved at {self.log_csv_path}')
