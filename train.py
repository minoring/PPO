import random

import yaml
import torch
import numpy as np
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


def criterion_actor(actor, actor_old, surrogate_objective, eps, obs, act, adv, ent_coef):
    # Note that log_prob compute separate log probabilities for each component of action.
    # However, we want to compute joint probability of action.
    # Thus, do summation.
    # log(p(a_1, a_2, a_3,... , a_n)) = log(p(a_1)) + ... , + log(p(a_n))
    # And our actions are independent due to our policy was diagonal Gaussian.
    actor_dist = actor(obs)
    actor_old_dist = actor_old(obs)
    logp = actor_dist.log_prob(act).sum(axis=-1)
    logp_old = actor_old_dist.log_prob(act).sum(axis=-1).detach()

    # Advantage normalization.
    # adv = (adv- adv.mean()) / (adv.std() + 1e-8)
    ratio = torch.exp(logp - logp_old)
    if surrogate_objective == 'clipping':
        loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)).mean()
    elif surrogate_objective == 'kl-penalty':
        kld = kl_divergence(actor_old_dist, actor_dist)
        loss = -((ratio * adv) - actor.beta * kld).mean()
    else:
        # Without penalty, clipping.
        loss = -(ratio * adv).mean()

    ent = actor_dist.entropy().mean()
    loss -= ent_coef * ent

    return loss, ent


def criterion_value(value_net, obs, ret, old_val, eps):
    val = value_net(obs)
    # Refer: https://github.com/openai/baselines/issues/91
    # But why clipped value not work...?
    # Clip the value to reduce variability during Critic training.
    # val_clipped = old_val + (val - old_val).clamp(-eps, eps)
    # val_loss1 = (val - ret).pow(2)
    # val_loss2 = (val_clipped - ret).pow(2)
    # return 0.5 * torch.max(val_loss1, val_loss2).mean()
    return 0.5 * (ret - val).pow(2).mean()


def main():
    print(f'Using device: {ptu.device}')
    args = get_train_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    hyperparams = config['hyperparams'][args.hyperparams]

    env = get_env(args)
    logger = Logger(args)

    # Setup random seed.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Initialize model, transition memory.
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = ActorNet(obs_dim, act_dim)
    actor_old = ActorNet(obs_dim, act_dim)
    value_net = ValueNet(obs_dim)
    actor.to(ptu.device)
    value_net.to(ptu.device)
    actor_old.to(ptu.device)
    memory = TransitionMemory(env, obs_dim, act_dim, args.n_actors, hyperparams)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=hyperparams['stepsize'])
    value_optim = torch.optim.Adam(value_net.parameters(), lr=hyperparams['stepsize'])
    timestep = 0  # The number of timestep in terms of environment interactions.
    num_updates = 0  # The number of epoch actor and critic are updated.
    while timestep < hyperparams['timestep']:
        timestep += memory.collect(actor, value_net)

        actor_old.load_state_dict(actor.state_dict())

        for epoch in range(hyperparams['num_epochs']):
            for minibatch in memory.minibatch_generator():
                obs, act, ret, adv, val = minibatch
                obs = obs.to(ptu.device)
                act = act.to(ptu.device)
                ret = ret.to(ptu.device)
                adv = adv.to(ptu.device)
                val = val.to(ptu.device)
                # Update policy.
                actor_loss, ent = criterion_actor(actor, actor_old, args.surrogate_objective,
                                                  hyperparams['eps'], obs, act, adv, args.ent_coef)
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                # Update value
                value_loss = criterion_value(value_net, obs, ret, val, hyperparams['eps'])
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()

        if args.surrogate_objective == 'kl-penalty':
            obs = memory.batch_obs[:-1].view(-1, obs_dim).to(ptu.device)
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
            avg_ep_ret = torch.sum(
                memory.batch_rew).item() / memory.num_ep  # Estimate episodic return
            print(f"Timestep: {timestep}, Average Episode Return: {avg_ep_ret}")
            logger.save_avg_ep_ret(avg_ep_ret, timestep)
            print("Actor Loss: {:.4f}\tValue Loss: {:.4f}".format(actor_loss.item(),
                                                                  value_loss.item()))
            logger.log_scalar('AvgEpRet', avg_ep_ret, timestep)
            logger.log_scalar('NumEpisodeInTrain', memory.num_ep, timestep)
            logger.log_scalar('ActorLoss', actor_loss.item(), timestep)
            logger.log_scalar('ValueLoss', value_loss.item(), timestep)
            logger.log_scalar('Entropy', ent.item(), timestep)
            logger.log_scalar('LogStd', actor.log_std.mean().item(), timestep)

    if args.model_save_path is not None:
        torch.save(actor.state_dict(), args.model_save_path)
        print(f'model saved at: {args.model_save_path}')

    logger.save()
    env.close()


if __name__ == '__main__':
    main()
