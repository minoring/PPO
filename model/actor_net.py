import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

import pytorch_utils as ptu


class ActorNet(nn.Module):
    """Proximal Policy Gradient"""
    def __init__(self, obs_dim, act_dim):
        super(ActorNet, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_out = nn.Linear(64, self.act_dim)
        log_std = torch.full((act_dim, ), -0.7, device=ptu.device)
        self.log_std = nn.Parameter(log_std)

        self.beta = 1 # KL-penalty coefficient.

    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(torch.float32)
        if len(obs.shape) <= 1:
            # If obs without batch dimension, expand.
            obs = obs.unsqueeze(0)

        obs = obs.to(ptu.device)
        with torch.no_grad():
            return self(obs).sample().detach().to('cpu')

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_out(x)
        std = torch.exp(self.log_std)

        return Normal(mean, std)
