import argparse


def get_train_args():
    parser = argparse.ArgumentParser(
        'Train an agent on high-dimensional continuous control problems')
    parser.add_argument('--seed',
                        help='random seed for gym, python, numpy, pytorch etc... (default: 1)',
                        type=int,
                        default=1)
    parser.add_argument('--config', '-c', help='Path to config file', default='config_ppo.yaml')
    parser.add_argument('--env',
                        '-e',
                        help='Environment for continuous control problem',
                        required=True)
    parser.add_argument('--render',
                        help='Whether render in test run',
                        action='store_true',
                        default=False)
    parser.add_argument('--hyperparams',
                        help='Hyperparameters to use. Check Appendix A. Schulman et al. 2017',
                        choices=['mujoco', 'humanoid', 'atari'],
                        required=True)
    parser.add_argument('--log-dir', help='Directory to save log', default='log')
    parser.add_argument('--log-csv-path', help='Path to csv file to save log', default='log.csv')
    parser.add_argument('--model-save-path', help='Path to save trained model')
    parser.add_argument('--log-interval',
                        help='How many batch update before log (default: 1)',
                        type=int,
                        default=1)
    parser.add_argument('--surrogate-objective',
                        help='Surrogate objective to train PPO',
                        choices=['no-clipping-penalty', 'clipping', 'kl-penalty'],
                        required=True)
    parser.add_argument('--ent-coef', help='Entropy coefficient', type=float, default=0.0)
    parser.add_argument('--n-actors',
                        help='The number of parallel actors to collect T timestpes of data',
                        type=int,
                        default=1)
    args = parser.parse_args()

    return args


def get_test_args():
    parser = argparse.ArgumentParser(
        'Test an agent on high-dimensional continuous control problems')
    parser.add_argument('--trained-model-path',
                        help='Path to trained model. Use this model for testing',
                        required=True)
    parser.add_argument('--env', '-e', help='Environment of RL algorithm', required=True)
    parser.add_argument('--record-video', help='Wheter to save video files', action='store_true')
    parser.add_argument('--render',
                        help='Whether render gym environment',
                        action='store_true',
                        default=False)
    parser.add_argument('--num-test-run', help='Number of test run', type=int, default=10)

    args = parser.parse_args()
    return args


def get_plot_args():
    parser = argparse.ArgumentParser('Plot learning curve')
    parser.add_argument('--learning-curve-csv',
                        help='Path to learning curve csv file of ppo',
                        required=True)
    parser.add_argument('--save-figure-path', help='Path to save the figure')

    args = parser.parse_args()
    return args
