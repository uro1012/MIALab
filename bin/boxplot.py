import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set list of repos manually
repo = ["2020-09-26-02-54-03", "2020-10-30-13-36-55"]

# set x labels
x_labels = ["ML", "Atlas"]


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    labels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']

    dice = [[0] * len(labels) for i in range(len(repo))]
    hdrfdst = [[0] * len(labels) for i in range(len(repo))]

    fig, axs = plt.subplots(2, 5)
    axs[0, 0].set(ylabel='Dice')
    axs[1, 0].set(ylabel='Hausdorff')

    for n in range(len(repo)):
        path = "mia-result/" + repo[n] + "/results.csv"
        results = pd.read_csv(path, sep=';')

        for i in range(len(labels)):
            dice[n][i] = results.loc[results['LABEL'] == labels[i]]['DICE'].values.tolist()
            hdrfdst[n][i] = results.loc[results['LABEL'] == labels[i]]['HDRFDST'].values.tolist()

    for i in range(len(labels)):
        axs[0, i].boxplot([d[i] for d in dice])
        axs[0, i].set_title(labels[i])
        axs[0, i].set_xticklabels(x_labels)

        axs[1, i].boxplot([d[i] for d in hdrfdst])
        axs[1, i].set_title(labels[i])
        axs[1, i].set_xticklabels(x_labels)

    plt.savefig("mia-result/" + repo[-1] + "/boxplot.png")
    plt.show()


if __name__ == '__main__':
    main()
