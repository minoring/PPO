import torch
import numpy as np

from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class TransitionMemory:
    def __init__(self, env, obs_dim, act_dim, n_actors, hyperparams):
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_actors = n_actors
        self.horizon = hyperparams['horizon']
        self.gamma = hyperparams['gamma']
        self.lambd = hyperparams['lambd']
        self.minibatch_size = hyperparams['minibatch_size']

        self.batch_size = self.n_actors * self.horizon
        # Initialize batch of transition.
        self.batch_obs = torch.zeros(self.horizon + 1, self.n_actors, self.obs_dim)
        self.batch_act = torch.zeros(self.horizon, self.n_actors, self.act_dim)
        self.batch_rew = torch.zeros(self.horizon, self.n_actors, 1)
        self.batch_val = torch.zeros(self.horizon + 1, n_actors, 1)
        self.batch_ret = torch.zeros(self.horizon + 1, self.n_actors, 1)
        self.batch_adv = torch.zeros(self.horizon, self.n_actors, 1)
        self.batch_not_done_mask = torch.full((self.horizon + 1, self.n_actors, 1),
                                              False,
                                              dtype=torch.bool)
        self.batch_obs[-1] = self.env.reset()  # Init first obs.

        self.num_ep = 0
        self.ep_rets = []  # Undiscounted returns for each episode. This is for logging.

    def collect(self, actor, critic):
        # Initialize from last trajectory.
        self.batch_obs[0].copy_(self.batch_obs[-1])
        self.batch_not_done_mask[0].copy_(self.batch_not_done_mask[-1])
        obs = self.batch_obs[0]
        # Collect Horizon x NumActors transition.
        for step in range(self.horizon):
            val = critic.get_value(obs)
            self.batch_val[step].copy_(val)
            act = actor.get_action(obs)
            obs, rew, done, _ = self.env.step(act)
            self.batch_obs[step + 1].copy_(obs)  # This is next observation.
            self.batch_act[step].copy_(act)
            self.batch_rew[step].copy_(rew)
            # The env.step() returns done in np.array type.
            # Convert it into torch tensor.
            done = torch.tensor(done[..., np.newaxis], dtype=torch.bool)
            not_done = torch.logical_not(done)
            # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
            # When using vectorized environments, the environments are automatically reset at the
            # end of each episode. Thus, the observation returned for the i-th environment when
            # done[i] is true will in fact be the first observation of the next episode,
            # not the last observation of the episode that has just terminated.
            self.batch_not_done_mask[step + 1].copy_(not_done)  # Whether next state is not done.

        self.num_ep = torch.logical_not(self.batch_not_done_mask).sum().item()
        # Compute Return and Advantage.
        next_val = critic.get_value(obs)
        self.batch_val[-1] = next_val
        self.batch_ret[-1] = next_val
        GAE = 0
        for step in reversed(range(self.horizon)):
            self.batch_ret[step] = self.batch_rew[step] + self.gamma * self.batch_ret[
                step + 1] * self.batch_not_done_mask[step + 1]
            delta = self.batch_rew[step] + self.gamma * self.batch_val[
                step + 1] * self.batch_not_done_mask[step + 1] - self.batch_val[step]
            GAE = delta + self.gamma * self.lambd * self.batch_not_done_mask[step + 1] * GAE
            self.batch_adv[step] = GAE

        return self.batch_size

    def minibatch_generator(self):
        # Random minibatch does not work... why?
        # sampler = BatchSampler(SubsetRandomSampler(range(self.batch_size)),
        #                        self.minibatch_size,
        #                        drop_last=True)
        sampler = BatchSampler(SequentialSampler(range(self.batch_size)),
                               self.minibatch_size,
                               drop_last=True)
        for indices in sampler:
            minibatch_obs = self.batch_obs[:-1].view(self.horizon * self.n_actors,
                                                     self.obs_dim)[indices]
            minibatch_act = self.batch_act.view(self.horizon * self.n_actors, self.act_dim)[indices]
            minibatch_ret = self.batch_ret[:-1].view(self.horizon * self.n_actors, 1)[indices]
            minibatch_adv = self.batch_adv.view(self.horizon * self.n_actors, 1)[indices]
            minibatch_val = self.batch_val[:-1].view(self.horizon * self.n_actors, 1)[indices]

            yield minibatch_obs, minibatch_act, minibatch_ret, minibatch_adv, minibatch_val

    def _compute_discount_sum(self, vals, discount):
        """
            input: vals = [a0, a1, a2]
            output: [a0 + discount * a1 + discount ^ 2 * a2,
                     a1 + discount * a2,
                     a2]
        """
        n = len(vals)
        discounted_sum = [0] * n
        discounted_sum[n - 1] = vals[n - 1]
        for i in reversed(range(n - 1)):
            discounted_sum[i] = vals[i] + discount * discounted_sum[i + 1]
        return discounted_sum
