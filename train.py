import random

import torch
import numpy as np
import torch.nn.functional as F

from parse_utils import get_train_args
from env import get_env
from model import ActorNet, ValueNet
from transition_memory import TransitionMemory
from logger import Logger
import pytorch_utils as ptu


def criterion_actor(actor, memory, value):
    adv_estimates = memory.batch_ret - value
    adv_estimates = ptu.to_tensor(adv_estimates)
    obs = ptu.to_tensor(memory.batch_obs)
    act = ptu.to_tensor(memory.batch_act)
    # Note that log_prob compute separate log probabilities for each component of action.
    # However, we want to compute joint probability of action.
    # Thus, do summation.
    # log(p(a_1, a_2, a_3,... , a_n)) = log(p(a_1)) + ... , + log(p(a_n))
    # And our actions are independent due to our policy was diagonal Gaussian.
    logp = actor(obs).log_prob(act).sum(axis=-1)
    return -(logp * adv_estimates).mean()


def criterion_value(value_net, memory):
    ret = ptu.to_tensor(memory.batch_ret)
    obs = ptu.to_tensor(memory.batch_obs)
    value = value_net(obs)

    return F.huber_loss(ret, value.squeeze(-1))


def get_episode_reward(env, actor):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        act = actor.get_action(obs)
        obs, reward, done, _ = env.step(act)
        episode_reward += reward
    return episode_reward


def main():
    args = get_train_args()
    env = get_env(args.env)
    logger = Logger(args.env, args.log_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    env.reset()
    actor = ActorNet(env.observation_space.shape[0], env.action_space.shape[0])
    value_net = ValueNet(env.observation_space.shape[0])
    actor.to(ptu.device)
    value_net.to(ptu.device)
    memory = TransitionMemory(args.batch_size, args.gamma)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=args.value_lr)
    for step in range(args.num_updates):
        memory.collect(env, actor)

        # Update value
        for _ in range(args.num_value_updates):
            value_loss = criterion_value(value_net, memory)
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

        # Update policy.
        value = value_net.get_value(memory.batch_obs)
        actor_loss = criterion_actor(actor, memory, value)
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        if step % args.log_interval == 0:
            episode_reward = get_episode_reward(env, actor)
            print(f"Step: {step}, Episode Reward: {episode_reward}")
            logger.ep_rewards.append(episode_reward)
            print("Actor Loss: {:.4f}\tValue Loss: {:.4f}".format(actor_loss.item(),
                                                                  value_loss.item()))
            logger.log_scalar('EpisodeReward', episode_reward, step)
            logger.log_scalar('ActorLoss', actor_loss.item(), step)
            logger.log_scalar('ValueLoss', value_loss.item(), step)

    torch.save(actor.state_dict(), args.model_save_path)
    print(f'model saved at: {args.model_save_path}')
    env.close()


if __name__ == '__main__':
    main()
