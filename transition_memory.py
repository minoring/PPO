import numpy as np


class TransitionMemory:
    def __init__(self, batch_size, gamma):
        self.batch_size = batch_size
        self.gamma = gamma
        self.batch_obs = []  # Should be converted into numpy array
        self.batch_act = []
        self.batch_ret = []
        self.num_ep = 0

    def collect(self, env, actor):
        # Clear previous batch
        self.batch_obs = []
        self.batch_act = []
        self.batch_ret = []
        self.num_ep = 0

        obs = env.reset()
        done = False
        while len(self.batch_obs) < self.batch_size:
            ep_obs = []
            ep_act = []
            ep_rew = []
            while not done:
                ep_obs.append(obs)
                act = actor.get_action(obs)
                ep_act.append(act)
                obs, rew, done, _ = env.step(act)
                ep_rew.append(rew)

            ep_ret = self._compute_discounted_reward_to_go(ep_rew)
            self.batch_obs.extend(ep_obs)
            self.batch_act.extend(ep_act)
            self.batch_ret.extend(ep_ret)
            obs = env.reset()
            done = False
            self.num_ep += 1

        self.batch_obs = np.array(self.batch_obs[:self.batch_size])
        self.batch_act = np.array(self.batch_act[:self.batch_size])
        self.batch_ret = np.array(self.batch_ret[:self.batch_size])

    def _compute_discounted_reward_to_go(self, ep_rew):
        n = len(ep_rew)
        reward_to_go = [0] * n
        reward_to_go[n - 1] = ep_rew[n - 1]
        for i in reversed(range(n - 1)):
            reward_to_go[i] = ep_rew[i] + self.gamma * reward_to_go[i + 1]
        return reward_to_go
