import enum
import os
import typing as t
import warnings

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt


def load_image(img_file, img_path='root', img_type='np'):
    if img_path == 'root':
        img_path = ''

    if img_type == 'np':
        img = np.load(os.path.join(img_path, img_file))
    elif img_type == 'sitk':
        img_sitk = sitk.ReadImage(os.path.join(img_path, img_file))
        img = sitk.GetArrayFromImage(img_sitk)

    return img


def display_slice(images, slice, single_plot='False', n_row_plot=1, n_col_plot=1, plane='axial'):
    fig, axs = plt.subplots(n_row_plot, n_col_plot)

    for i in range(len(images)):
        if plane=='axial':
            image_2d = images[i][slice, :, :]

        elif plane=='coronal':
            image_2d = images[i][:, slice, :]

        elif plane=='sagital':
            image_2d = images[i][:, :, slice]

        x_index = i % n_col_plot
        y_index = int(np.floor(i/n_col_plot))

        if n_row_plot > 1 and n_col_plot > 1:
            axs[x_index, y_index].imshow(image_2d, interpolation='nearest')
            axs[x_index, y_index].axis('equal')
            axs[x_index, y_index].axis("off")

        elif n_row_plot == 1:
            axs[x_index].imshow(image_2d, interpolation='nearest')
            axs[x_index].axis('equal')
            axs[x_index].axis("off")

        elif n_col_plot == 1:
            axs[y_index].imshow(image_2d, interpolation='nearest')
            axs[y_index].axis('equal')
            axs[x_index].axis("off")

        plt.draw()

    plt.show()

    return fig, axs


def display_3planes(image, intersect_point):
    x_max, y_max, z_max = image.shape

    axial_img = image[intersect_point[0], :, :]
    coronal_img = image[:, intersect_point[1], :]
    sagital_img = image[:, :, intersect_point[2]]

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(axial_img, interpolation='nearest')
    axs[0, 0].axis('equal')
    axs[0, 0].set_title('Axial plane', fontsize=10)
    axs[0, 0].axis("off")

    axs[0, 1].imshow(coronal_img, interpolation='nearest')
    axs[0, 1].axis('equal')
    axs[0, 1].set_title('Coronal plane', fontsize=10)
    axs[0, 1].axis("off")

    axs[1, 0].imshow(sagital_img, interpolation='nearest')
    axs[1, 0].axis('equal')
    axs[1, 0].set_title('Sagital plane', fontsize=10)
    axs[1, 0].axis("off")

    plt.draw()
    plt.show()

    return fig, axs


def main():
    # Display 3 plane image
    img_sba_affine = load_image('../data/atlas/atlas_prediction_SBA_affine.npy')
    display_3planes(img_sba_affine, [100, 120, 50])

    # Display the same slice of two different images
    img_mj_affine = load_image('../data/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz', img_type='sitk')
    img_list = [img_mj_affine, img_sba_affine]
    display_slice(img_list, 100, n_row_plot=1, n_col_plot=2)


if __name__ == '__main__':
    main()
