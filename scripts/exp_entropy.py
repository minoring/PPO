"""Experiment the effect of entropy using Walker2d env"""
import os
import subprocess

seeds = [1, 2, 3]
ent_coefs = [0.0, 0.2]


def main():
    cmd = "python train.py --env Walker2d-v2 --hyperparams mujoco --log-interval 1 --surrogate-objective clipping --log-csv-path entropy.csv"
    for seed in seeds:
        for ent_coef in ent_coefs:
            run = cmd + ' --seed=' + str(seed) + ' --ent-coef=' + str(ent_coef)

            subprocess.check_call(run, env=os.environ, shell=True)


if __name__ == '__main__':
    main()
