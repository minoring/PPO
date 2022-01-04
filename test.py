import torch
import numpy as np
import gym
from gym.wrappers import RecordVideo

from model import ActorNet
from parse_utils import get_test_args
import pytorch_utils as ptu
import pytorch_utils as ptu


def main():
    args = get_test_args()

    env = gym.make(args.env)

    actor = ActorNet(env.observation_space.shape[0], env.action_space.shape[0])
    actor.load_state_dict(torch.load(args.trained_model_path))
    actor.to(ptu.device)

    epsiode_rewards = []
    for test_run in range(args.num_test_run):
        # Save video if it is last test run.
        if args.record_video and test_run == args.num_test_run - 1:
            video_save_folder = f'{args.env}_video'
            env = RecordVideo(env, video_save_folder)
            print(f'video will be saved at: {video_save_folder}')

        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if args.render:
                env.render()
            act = actor.get_action(obs)
            obs, reward, done, _ = env.step(act)
            episode_reward += reward

        print(f"Episode reward test run {test_run}: {episode_reward}")
        epsiode_rewards.append(episode_reward)

    print(f'Average score for {args.num_test_run} reward: {np.mean(epsiode_rewards)}')
    print(f'Standard deviation score for {args.num_test_run} reward: {np.std(epsiode_rewards)}')


if __name__ == '__main__':
    main()
