import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    labels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']
    path = "mia-result/2020-09-26-02-54-03/results.csv"
    results = pd.read_csv(path, sep=';')

    shape = (len(labels), int(len(results)/len(labels)))
    dice = np.zeros(shape)
    hdrfdst = np.zeros(shape)
    for i in range(len(labels)):
        dice[i] = results.loc[results['LABEL'] == labels[i]]['DICE'].values.tolist()
        hdrfdst[i] = results.loc[results['LABEL'] == labels[i]]['HDRFDST'].values.tolist()

        plt.subplot(2, len(labels), i+1)
        plt.boxplot(dice[i])
        plt.title(labels[i])
        if i == 0:
            plt.ylabel('Dice')

        plt.subplot(2, len(labels), i+1+len(labels))
        plt.boxplot(hdrfdst[i])
        plt.title(labels[i])
        if i == 0:
            plt.ylabel('Hausdorff')

    plt.show()


if __name__ == '__main__':
    main()
