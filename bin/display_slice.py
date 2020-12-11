import enum
import os
import typing as t
import warnings

import numpy as np
from scipy import stats
from scipy import special
from sklearn.utils.extmath import weighted_mode
from scipy.ndimage.morphology import distance_transform_edt
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk
from matplotlib import pyplot as plt

def display_slice(images, slice, single_plot=1):
    fig = plt.figure(figsize=(8, 8))

    n_row_plot = 5
    n_col_plot = 2

    if single_plot:
        image_3d_np = sitk.GetArrayFromImage(images)
        image_2d_np = image_3d_np[slice, :, :]

        plt.imshow(image_2d_np, interpolation='nearest')
        plt.draw()
    else:
        for i in range(1, len(images)+1):
            fig.add_subplot(n_row_plot, n_col_plot, i)
            image_3d_np = sitk.GetArrayFromImage(images[i-1])
            image_2d_np = image_3d_np[slice, :, :]

            plt.imshow(image_2d_np, interpolation='nearest')
            plt.draw()
    plt.show()

def display_3planes(image, intersect_point):
    axial_img = image[intersect_point[0], :, :]
    coronal_img = image[:, intersect_point[1], :]
    sagital_img = image[:, :, intersect_point[2]]

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs[0].plot(axial_img, interpolation='nearest')
    axs[0].set_title('axial plane')
    axs[1].imshow(coronal_img, interpolation='nearest')
    axs[1].set_title('coronal plane')
    axs[2].imshow(sagital_img, interpolation='nearest')
    axs[2].set_title('sagital plane')

    plt.draw()
    plt.show()


def main():
    img = np.load('../data/atlas/atlas_prediction_SBA_affine.npy')
    display_3planes(img, [100, 120, 50])


if __name__ == '__main__':
    main()
