import os
import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, args):
        self.env_name = args.env
        self.log_dir = os.path.join(args.log_dir,
                                    self.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.avg_ep_rets = []
        self.total_env_interacts = []
        self.seed = args.seed
        self.log_csv_path = args.log_csv_path
        self.surrogate_objective = args.surrogate_objective
        self.ent_coef = args.ent_coef
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, scalar, step):
        self.writer.add_scalar(name, scalar, step)

    def save_avg_ep_ret(self, ep_ret, env_interact):
        self.avg_ep_rets.append(ep_ret)
        self.total_env_interacts.append(env_interact)

    def save(self):
        env_names = [self.env_name] * len(self.avg_ep_rets)
        ep_idxs = list(range(1, len(self.avg_ep_rets) + 1))
        seeds = [self.seed] * len(self.avg_ep_rets)
        surr_objs = [self.surrogate_objective] * len(self.avg_ep_rets)
        ent_coefs = [self.ent_coef] * len(self.avg_ep_rets)

        df = pd.DataFrame(list(
            zip(ep_idxs, env_names, self.total_env_interacts, self.avg_ep_rets, surr_objs,
                ent_coefs, seeds)),
                          columns=[
                              'Episode', 'Env', 'TotalEnvInteracts', 'AvgEpRet', 'SurrObj',
                              'EntCoef', 'Seed'
                          ])

        df.to_csv(self.log_csv_path,
                  mode='a',
                  header=not os.path.exists(self.log_csv_path),
                  index=False)
        print(f'Log saved at {self.log_csv_path}')
