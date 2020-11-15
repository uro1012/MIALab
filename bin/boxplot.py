import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set list of repos manually
repo = ["AB_Affine", "AB_Non_Rigid", "ML_Affine", "ML_Non_Rigid"]

# set x labels
x_labels = ["Atlas Aff", "Atlas NR", "ML Aff", "ML NR"]


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
    axs[0, 0].set_ylabel('Dice', fontsize=20)
    axs[1, 0].set_ylabel('Hausdorff', fontsize=20)

    for n in range(len(repo)):
        path = "mia-result/" + repo[n] + "/results.csv"
        results = pd.read_csv(path, sep=';')

        for i in range(len(labels)):
            dice[n][i] = results.loc[results['LABEL'] == labels[i]]['DICE'].values.tolist()
            hdrfdst[n][i] = results.loc[results['LABEL'] == labels[i]]['HDRFDST'].values.tolist()

    for i in range(len(labels)):
        axs[0, i].boxplot([d[i] for d in dice])
        axs[0, i].set_title(labels[i], fontsize=16)
        axs[0, i].set_xticklabels(x_labels, rotation=45, fontsize=10)

        axs[1, i].boxplot([d[i] for d in hdrfdst])
        axs[1, i].set_xticklabels(x_labels, rotation=45, fontsize=10)

    plt.savefig("mia-result/boxplot.png")
    plt.show()


if __name__ == '__main__':
    main()
