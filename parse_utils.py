import argparse


def get_train_args():
    parser = argparse.ArgumentParser(
        'Train an agent on high-dimensional continuous control problems')
    parser.add_argument('--seed',
                        help='random seed for gym, python, numpy, pytorch etc... (default: 1)',
                        type=int,
                        default=1)
    parser.add_argument('--env',
                        '-e',
                        help='Environment for continuous control problem',
                        required=True)
    parser.add_argument('--batch-size',
                        help='Batch size to update policy one step (default: 4096)',
                        type=int,
                        default=4096)
    parser.add_argument('--log-dir', help='Directory to save log', default='log')
    parser.add_argument('--model-save-path', help='Path to save trained model', default='model.pt')
    parser.add_argument('--gamma', help='Discount factor (default: 0.99)', type=float, default=0.99)
    parser.add_argument('--num-updates',
                        help='Number of update steps (default: 3000)',
                        type=int,
                        default=5000)
    parser.add_argument('--log-interval',
                        help='How many steps to update before log (default: 10)',
                        type=int,
                        default=10)
    parser.add_argument('--num-value-updates',
                        help='Number of update stpes for value estimates (default: 20)',
                        type=int,
                        default=40)
    parser.add_argument('--actor-lr',
                        help='Learning rate for actor (default: 3e-4)',
                        type=float,
                        default=3e-4)
    parser.add_argument('--value-lr',
                        help='Learning rate for value function(default: 3e-4)',
                        type=float,
                        default=3e-4)
    parser.add_argument('--surrogate-objective',
                        help='Surrogate objective to train PPO',
                        choices=['no-clipping-penalty', 'clipping', 'kl-penalty'],
                        required=True)
    args = parser.parse_args()

    return args


def get_test_args():
    parser = argparse.ArgumentParser("Test DQN playing Atari")
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
