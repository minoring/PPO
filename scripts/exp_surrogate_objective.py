"""Experiment surrogate objective in various environments"""
import os
import subprocess

envs = [
    'HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2',
    'Swimmer-v2', 'Walker2d-v2'
]
surrogate_objectives = ['clipping', 'kl-penalty', 'no-clipping-penalty']
seeds = [1, 2, 3]


def main():
    cmd = "python train.py --hyperparams mujoco --log-interval 1 --log-csv-path surrogate_objective.csv --ent-coef 0.2"
    for seed in seeds:
        for obj in surrogate_objectives:
            for env in envs:
                run = cmd + ' --seed=' + str(
                    seed) + ' --env=' + env + ' --surrogate-objective=' + obj

                subprocess.check_call(run, env=os.environ, shell=True)


if __name__ == '__main__':
    main()
