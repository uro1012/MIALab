import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set list of repos manually
repo = ["ML_Affine_D40_E2", "ML_Affine_D40_E5", "ML_Affine_D40_E10", "ML_Affine_D40_E20"]

# set x labels
x_ticks = [2, 5, 10, 20]


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    labels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']

    dice = [[0] * len(labels) for i in range(len(repo))]
    hdrfdst = [[0] * len(labels) for i in range(len(repo))]

    fig, axs = plt.subplots(2, 5, figsize=(16, 10))
    fig.suptitle('Machine Learning optimization Parameter Estimator\nwith constant Depth of 40', fontsize=20)
    axs[0, 0].set_ylabel('Dice', fontsize=20)
    axs[1, 0].set_ylabel('Hausdorff', fontsize=20)

    for n in range(len(repo)):
        path = "mia-result/" + repo[n] + "/results.csv"
        results = pd.read_csv(path, sep=';')

        for i in range(len(labels)):
            dice[n][i] = np.mean(results.loc[results['LABEL'] == labels[i]]['DICE'].values.tolist())
            hdrfdst[n][i] = np.mean(results.loc[results['LABEL'] == labels[i]]['HDRFDST'].values.tolist())

    for i in range(len(labels)):
        axs[0, i].plot(x_ticks, [d[i] for d in dice],'r-+')
        axs[0, i].set_ylim(0, 1)
        axs[0, i].set_title(labels[i], fontsize=16)
        axs[0, i].set_xticks(x_ticks)

        axs[1, i].plot(x_ticks, [h[i] for h in hdrfdst], 'r-+')
        axs[1, i].set_ylim(0, np.max(hdrfdst))
        axs[1, i].set_xticks(x_ticks)

    plt.savefig("mia-result/plot.png")
    plt.show()


if __name__ == '__main__':
    main()
