import gym
from gym.spaces import Box


def get_env(env):
    env = gym.make(env)

    assert isinstance(env.action_space,
                      Box), "This project works for envs with continuous action spaces."
    return env
