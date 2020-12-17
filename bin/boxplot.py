import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def boxplot(repos, x_labels, title="", filename="boxplot", show=False):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    labels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']

    dice = [[0] * len(labels) for i in range(len(repos))]
    hdrfdst = [[0] * len(labels) for i in range(len(repos))]

    fig, axs = plt.subplots(2, 5, figsize=(16, 10))
    fig.suptitle(title, fontsize=30)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    axs[0, 0].set_ylabel('Dice', fontsize=24)
    axs[1, 0].set_ylabel('Hausdorff', fontsize=24)

    for n in range(len(repos)):
        path = "mia-result/" + repos[n] + "/results.csv"
        results = pd.read_csv(path, sep=';')

        for i in range(len(labels)):
            dice[n][i] = results.loc[results['LABEL'] == labels[i]]['DICE'].values.tolist()
            hdrfdst[n][i] = results.loc[results['LABEL'] == labels[i]]['HDRFDST'].values.tolist()

    for i in range(len(labels)):
        axs[0, i].boxplot([d[i] for d in dice])
        axs[0, i].set_ylim(0, 1)
        axs[0, i].set_title(labels[i], fontsize=24)
        axs[0, i].set_xticklabels(x_labels, rotation=45, fontsize=16)
        axs[0, i].tick_params(labelsize=16)

        axs[1, i].boxplot([d[i] for d in hdrfdst])
        axs[1, i].set_ylim(0, np.max(hdrfdst))
        axs[1, i].set_xticklabels(x_labels, rotation=45, fontsize=16)
        axs[1, i].tick_params(labelsize=16)

    plt.savefig("mia-result/" + filename + ".png")
    if show:
        plt.show()


def main():
    reposNR = ["AB_Affine_MV", "AB_Affine_GW", "AB_Affine_LW", "AB_Affine_SBA", "ML_Affine"]
    x_labelsNR = ["MV", "GW", "LW", "SBA", "ML"]
    boxplot(reposNR, x_labelsNR, "Comparaison of affine segmentations", "boxplot_Affine_all")

    reposAffine = ["AB_Non_Rigid_MV", "AB_Non_Rigid_GW", "AB_Non_Rigid_LW", "AB_Non_Rigid_SBA", "ML_Non_Rigid"]
    x_labelsAffine = ["MV", "GW", "LW", "SBA", "ML"]
    boxplot(reposAffine, x_labelsAffine, "Comparaison of non-rigid segmentations", "boxplot_NR_all")

    reposComp = ["AB_Non_Rigid_MV", "AB_Affine_MV", "ML_Non_Rigid", "ML_Affine"]
    x_labelsComp = ["MV NR", "MV Aff", "ML NR", "ML Aff"]
    boxplot(reposComp, x_labelsComp, "Comparaison between non-rigid and affine registration", "boxplotComparisonNRAff")


if __name__ == '__main__':
    main()
