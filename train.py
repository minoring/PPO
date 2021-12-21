import random

import yaml
import torch
import numpy as np
import torch.nn.functional as F

from parse_utils import get_train_args
from env import get_env
from model import ActorNet, ValueNet
from transition_memory import TransitionMemory
from logger import Logger
import pytorch_utils as ptu


def criterion_actor(actor, actor_old, surrogate_objective, hyperparams, obs, act, adv):
    obs = ptu.to_tensor(obs)
    act = ptu.to_tensor(act)
    adv = ptu.to_tensor(adv)
    # Note that log_prob compute separate log probabilities for each component of action.
    # However, we want to compute joint probability of action.
    # Thus, do summation.
    # log(p(a_1, a_2, a_3,... , a_n)) = log(p(a_1)) + ... , + log(p(a_n))
    # And our actions are independent due to our policy was diagonal Gaussian.
    if surrogate_objective == 'clipping':

    logp = actor(obs).log_prob(act).sum(axis=-1)
    return -(logp * adv).mean()


def criterion_value(value_net, obs, ret):
    obs = ptu.to_tensor(obs)
    ret = ptu.to_tensor(ret)
    value = value_net(obs)

    return F.mse_loss(ret, value.squeeze(-1))


def get_episode_reward(env, render, actor):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        act = actor.get_action(obs)
        obs, reward, done, _ = env.step(act)
        if render:
            env.render()
        episode_reward += reward
    return episode_reward


def main():
    args = get_train_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    hyperparams = config['hyperparams'][args.hyperparams]

    env = get_env(args.env)
    logger = Logger(args.env, args.log_dir, args.log_csv_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    env.reset()
    actor = ActorNet(env.observation_space.shape[0], env.action_space.shape[0])
    actor_old = ActorNet(env.observation_space.shape[0], env.action_space.shape[0])
    value_net = ValueNet(env.observation_space.shape[0])
    actor.to(ptu.device)
    value_net.to(ptu.device)
    actor_old.to(ptu.device)
    memory = TransitionMemory(hyperparams['horizon'], hyperparams['gamma'], hyperparams['lambd'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=hyperparams['stepsize'])
    value_optim = torch.optim.Adam(value_net.parameters(), lr=hyperparams['stepsize'])
    timestep = 0
    num_updates = 0
    while timestep < hyperparams['timestep']:
        timestep += memory.collect(env, actor, value_net)

        actor_old.load_state_dict(actor.state_dict())
        for epoch in range(hyperparams['num_epochs']):
            # memory.shuffle()
            for minibatch_idx in range(len(memory.batch_obs) // hyperparams['minibatch_size']):
                obs, act, ret, adv = memory.get_minibatch(minibatch_idx,
                                                          hyperparams['minibatch_size'])
                # Update policy.
                actor_loss = criterion_actor(actor, actor_old, args.surrogate_objective, hyperparams, obs, act, adv)
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                # Update value
                value_loss = criterion_value(value_net, obs, ret)
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()

        num_updates += 1
        if num_updates % args.log_interval == 0:
            episode_reward = get_episode_reward(env, args.render, actor)
            print(f"Timestep: {timestep}, Episode Reward: {episode_reward}")
            logger.ep_rewards.append(episode_reward)
            print("Actor Loss: {:.4f}\tValue Loss: {:.4f}".format(actor_loss.item(),
                                                                  value_loss.item()))
            logger.log_scalar('EpisodeReward', episode_reward, timestep)
            logger.log_scalar('ActorLoss', actor_loss.item(), timestep)
            logger.log_scalar('ValueLoss', value_loss.item(), timestep)

    torch.save(actor.state_dict(), args.model_save_path)
    print(f'model saved at: {args.model_save_path}')
    logger.save()
    env.close()


if __name__ == '__main__':
    main()
