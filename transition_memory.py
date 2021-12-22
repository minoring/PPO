from multiprocessing import Process

import numpy as np


class NTransitionMemory:
    # TODO(minho): Implement N actor collect.
    def __init__(self, num_actors, horizon, gamma, lambd):
        self.num_actors = num_actors
        self.transition_memories = [
            TransitionMemory(horizon, gamma, lambd) for _ in range(num_actors)
        ]

    def collect(self, env, actor, critic):
        procs = []
        for i in range(self.num_actors):
            proc = Process(target=self.transition_memories[i].collect, args=(env, actor, critic))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()


class TransitionMemory:
    def __init__(self, horizon, gamma, lambd):
        self.horizon = horizon
        self.gamma = gamma
        self.lambd = lambd
        self.clear_batch()
        self.num_ep = 0

    def collect(self, env, actor, critic):
        self.clear_batch()
        self.num_ep = 0

        obs = env.reset()
        done = False
        ep_rew = []
        ep_val = []
        for i in range(self.horizon):
            ep_val.append(critic.get_value(obs))
            self.batch_obs.append(obs)
            act = actor.get_action(obs)
            self.batch_act.append(act)
            obs, rew, done, _ = env.step(act)
            ep_rew.append(rew)

            if done:
                self.batch_ret.extend(self._compute_discount_sum(ep_rew, self.gamma))

                # Compute Advantage.
                # High-Dimensional Continuous Control Using Generalized Advantage Estimation, John Schulman et al. 2016
                ep_val.append(0)  # Add dummy zero for indexing.
                ep_val = np.array(ep_val)
                deltas = ep_rew + self.gamma * ep_val[1:] - ep_val[:-1]
                self.batch_adv.extend(self._compute_discount_sum(deltas, self.gamma * self.lambd))

                # Reset the environment.
                ep_rew = []
                ep_val = []
                obs = env.reset()
                done = False
                self.num_ep += 1
            elif i == self.horizon - 1:
                # Bootstrap from next state.
                next_obs_value = critic.get_value(obs)
                ep_rew[-1] = next_obs_value
                self.batch_ret.extend(self._compute_discount_sum(ep_rew, self.gamma))

                # Compute Advantage.
                ep_val.append(next_obs_value)
                ep_val = np.array(ep_val)
                deltas = ep_rew + self.gamma * ep_val[1:] - ep_val[:-1]
                self.batch_adv.extend(self._compute_discount_sum(deltas, self.gamma * self.lambd))
                self.num_ep += 1

        assert len(self.batch_obs) == self.horizon
        assert len(self.batch_act) == self.horizon
        assert len(self.batch_ret) == self.horizon
        assert len(self.batch_adv) == self.horizon

        self.batch_obs = np.array(self.batch_obs)
        self.batch_act = np.array(self.batch_act)
        self.batch_ret = np.array(self.batch_ret)
        self.batch_adv = np.array(self.batch_adv)

        return len(self.batch_obs)

    def get_minibatch(self, minibatch_idx, minibatch_size):
        obs = self.batch_obs[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]
        act = self.batch_act[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]
        ret = self.batch_ret[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]
        adv = self.batch_adv[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]
        return obs, act, ret, adv

    def shuffle(self):
        """Shuffle the batch so that minibatch do not always optimize same thing"""
        idx = np.random.permutation(len(self.batch_obs))
        self.batch_obs = self.batch_obs[idx]
        self.batch_act = self.batch_act[idx]
        self.batch_ret = self.batch_ret[idx]
        self.batch_adv = self.batch_adv[idx]

    def clear_batch(self):
        self.batch_obs = []
        self.batch_act = []
        self.batch_ret = []
        self.batch_adv = []

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
