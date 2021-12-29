"""Train the model to get saved model for each environment."""
import os
import subprocess

envs = [
    'HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2',
    'Swimmer-v2', 'Walker2d-v2'
]


def main():
    cmd = "python train.py --hyperparams mujoco --log-interval 1  --surrogate-objective clipping --ent-coef 0.2 --seed 1"
    for env in envs:
        run = cmd + ' --env=' + env + ' --model-save-path=' + env + '.pt'

        subprocess.check_call(run, env=os.environ, shell=True)


if __name__ == '__main__':
    main()
