import torch
import torch.nn as nn

import pytorch_utils as ptu


class ValueNet(nn.Module):
    """Compute value function given state"""
    def __init__(self, obs_dim):
        super(ValueNet, self).__init__()
        self.obs_dim = obs_dim
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        x = self.out(x)
        return x

    def get_value(self, obs):
        if len(obs.shape) <= 1:
            # If obs without batch dimension, expand.
            obs = obs.unsqueeze(0)

        obs = obs.to(ptu.device)
        with torch.no_grad():
            return self(obs).detach().to('cpu')
