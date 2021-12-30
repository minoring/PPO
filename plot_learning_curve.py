import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import pandas as pd
import numpy as np

from parse_utils import get_plot_args

ENVS = [
    'HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2',
    'Swimmer-v2', 'Walker2d-v2'
]


def main():
    args = get_plot_args()
    sns.set(style='darkgrid')

    df = pd.read_csv(args.learning_curve_csv)
    n_row = 2
    n_col = 4
    fig, axes = plt.subplots(n_row, n_col, figsize=(28, 10))
    df = df.rename(columns={'SurrObj': 'surrogate objective'})
    g = None
    for r in range(n_row):
        for c in range(n_col):
            if r == n_row - 1 and c == n_col - 1:
                axes[r][c].legend('upper left')
                break
            env = ENVS[r * n_col + c]
            axes[r][c].set_title(env)

            if g is not None:
                # Remove the legend in the axes before.
                g.legend().remove()
            g = sns.lineplot(x='TotalEnvInteracts',
                             y='AvgEpRet',
                             hue='surrogate objective',
                             estimator='mean',
                             ci='sd',
                             palette=sns.color_palette("husl", 3),
                             data=df,
                             ax=axes[r][c])
            sns.move_legend(g, 'upper left')
            axes[r][c].get_legend().set_title('')
            axes[r][c].set_xlabel('')
            axes[r][c].set_ylabel('')
            axes[r][c].ticklabel_format(axis='x', style='plain')
    plt.show()
    if args.save_figure_path is not None:
        fig.savefig(args.save_figure_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
