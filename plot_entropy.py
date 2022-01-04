import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from parse_utils import get_plot_args

ENV = 'Walker2d-v2'


def main():
    args = get_plot_args()
    sns.set(style='darkgrid')

    df = pd.read_csv(args.learning_curve_csv)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    df_renamed = df.rename(columns={'EntCoef': 'entropy'})
    df_renamed.loc[np.isclose(df['EntCoef'], 0.0), 'entropy'] = 'without_entropy'
    df_renamed.loc[np.logical_not(np.isclose(df['EntCoef'], 0.0)), 'entropy'] = 'with_entropy'

    ax.set_title(ENV)
    g = sns.lineplot(x='TotalEnvInteracts',
                     y='AvgEpRet',
                     hue='entropy',
                     estimator='mean',
                     ci='sd',
                     palette=sns.color_palette("Set2", 2),
                     data=df_renamed,
                     ax=ax)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Return')
    plt.ticklabel_format(axis='x', style='plain')
    plt.show()


if __name__ == '__main__':
    main()
