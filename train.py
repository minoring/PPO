import random

import yaml
import torch
import numpy as np
import torch.nn.functional as F
from torch.linalg import det, inv

from parse_utils import get_train_args
from env import get_env
from model import ActorNet, ValueNet
from transition_memory import TransitionMemory
from logger import Logger
import pytorch_utils as ptu


def kl_divergence(dist1, dist2):
    """KL divergence between two diagonal Gaussian distribution"""
    b = dist1.loc.shape[0]  # Batch size.
    k = dist1.loc.shape[-1]  # k dimensional Gaussian.
    mu1 = dist1.loc
    mu2 = dist2.loc
    sigma1 = torch.diag_embed(dist1.scale)
    sigma2 = torch.diag_embed(dist2.scale)

    # How to compute KL divergence between two multivariate Gaussian.
    # Refer: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    kld = 0.5 * (torch.log(det(sigma2) / det(sigma1)) - k + torch.bmm(
        (mu1 - mu2).view(b, 1, k), torch.bmm(inv(sigma2), (mu1 - mu2).view(b, k, 1))) +
                 torch.bmm(inv(sigma2), sigma1).diagonal(dim1=1, dim2=2).sum(dim=-1))
    return kld


def criterion_actor(actor, actor_old, surrogate_objective, hyperparams, obs, act, adv, ent_coef):
    obs = ptu.to_tensor(obs)
    act = ptu.to_tensor(act)
    adv = ptu.to_tensor(adv)
    # Note that log_prob compute separate log probabilities for each component of action.
    # However, we want to compute joint probability of action.
    # Thus, do summation.
    # log(p(a_1, a_2, a_3,... , a_n)) = log(p(a_1)) + ... , + log(p(a_n))
    # And our actions are independent due to our policy was diagonal Gaussian.
    actor_dist = actor(obs)
    actor_old_dist = actor_old(obs)
    logp = actor_dist.log_prob(act).sum(axis=-1)
    logp_old = actor_old_dist.log_prob(act).sum(axis=-1).detach()
    kld = kl_divergence(actor_old_dist, actor_dist)

    ratio = torch.exp(logp - logp_old)
    if surrogate_objective == 'clipping':
        loss = -(torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - hyperparams['eps'], 1 + hyperparams['eps']) * adv)).mean()
    elif surrogate_objective == 'kl-penalty':
        loss = -((ratio * adv) - actor.beta * kld).mean()
    else:
        # Without penalty, clipping.
        loss = -(ratio * adv).mean()

    ent = actor_dist.entropy().mean()
    loss -= ent_coef * ent

    return loss, ent


def criterion_value(value_net, obs, ret):
    obs = ptu.to_tensor(obs)
    ret = ptu.to_tensor(ret)
    value = value_net(obs)

    return F.mse_loss(ret, value.squeeze(-1))


def main():
    args = get_train_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    hyperparams = config['hyperparams'][args.hyperparams]

    env = get_env(args.env)
    logger = Logger(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

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
        # Time
        timestep += memory.collect(env, actor, value_net)

        actor_old.load_state_dict(actor.state_dict())

        for epoch in range(hyperparams['num_epochs']):
            if args.shuffle:
                memory.shuffle()
            for minibatch_idx in range(len(memory.batch_obs) // hyperparams['minibatch_size']):
                obs, act, ret, adv = memory.get_minibatch(minibatch_idx,
                                                          hyperparams['minibatch_size'])
                # Update policy.
                actor_loss, ent = criterion_actor(actor, actor_old, args.surrogate_objective,
                                                  hyperparams, obs, act, adv, args.ent_coef)
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                # Update value
                value_loss = criterion_value(value_net, obs, ret)
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()

        if args.surrogate_objective == 'kl-penalty':
            obs = ptu.to_tensor(memory.batch_obs)
            kld = kl_divergence(actor_old(obs), actor(obs)).mean().item()
            if kld < hyperparams['kl_target'] / 1.5:
                actor.beta /= 2
            elif kld > hyperparams['kl_target'] * 1.5:
                actor.beta *= 2

            if num_updates % args.log_interval == 0:
                logger.log_scalar('KLDivergence', kld, timestep)
                logger.log_scalar('Beta', actor.beta, timestep)

        num_updates += 1
        if num_updates % args.log_interval == 0:
            avg_ep_ret = np.mean(memory.ep_rets)
            print(f"Timestep: {timestep}, Average Episode Return: {avg_ep_ret}")
            logger.save_avg_ep_ret(avg_ep_ret, timestep)
            print("Actor Loss: {:.4f}\tValue Loss: {:.4f}".format(actor_loss.item(),
                                                                  value_loss.item()))
            logger.log_scalar('AvgEpRet', avg_ep_ret, timestep)
            logger.log_scalar('NumEpisodeInTrain', memory.num_ep, timestep)
            logger.log_scalar('ActorLoss', actor_loss.item(), timestep)
            logger.log_scalar('ValueLoss', value_loss.item(), timestep)
            logger.log_scalar('Entropy', ent.item(), timestep)

    if args.model_save_path is not None:
        torch.save(actor.state_dict(), args.model_save_path)
        print(f'model saved at: {args.model_save_path}')

    logger.save()
    env.close()


if __name__ == '__main__':
    main()
