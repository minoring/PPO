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
        self.std = torch.full((act_dim, ), 0.1,
                              device=ptu.device)  # Standard deviations are not trained.

    def get_action(self, obs):
        assert len(obs.shape) <= 1 # Single observation is expected as an input.
        obs = obs[np.newaxis, ...]
        obs = ptu.to_tensor(obs)
        with torch.no_grad():
            return ptu.to_numpy(self(obs).sample().squeeze(0))

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_out(x)

        return Normal(mean, self.std)
