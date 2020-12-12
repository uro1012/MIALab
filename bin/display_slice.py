import enum
import os
import typing as t
from tkinter import filedialog
from sklearn import preprocessing

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
    coronal_img = image[::-1, intersect_point[1], :]
    sagital_img = image[::-1, :, intersect_point[2]]


    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    vmax = np.max(image)

    ax1.imshow(axial_img, interpolation='nearest', vmax=vmax, vmin=0)
    ax1.axis('equal')
    ax1.set_title('Axial plane', fontsize=10)
    ax1.axis("off")

    ax2.imshow(coronal_img, interpolation='nearest', vmax=vmax, vmin=0)
    ax2.axis('equal')
    ax2.set_title('Coronal plane', fontsize=10)
    ax2.axis("off")

    ax3.imshow(sagital_img, interpolation='nearest', vmax=vmax, vmin=0)
    ax3.axis('equal')
    ax3.set_title('Sagital plane', fontsize=10)
    ax3.axis("off")

    plt.draw()
    plt.show()

    return


def main():
    # Display 3 plane image
    filepath = filedialog.askopenfilename()
    if filepath[-3:] == 'npy':
        img = load_image(filepath)
    else:
        img = load_image(filepath, img_type='sitk')
    display_3planes(img, [100, 120, 50])

    # Display the same slice of two different images
    files = filedialog.askopenfilename(multiple=True)
    imgs = []
    for file in files:
        if file[-3:] == 'npy':
            imgs.append(load_image(file))
        else:
            imgs.append(load_image(file, img_type='sitk'))
    display_slice(imgs, 100, n_row_plot=2, n_col_plot=2)


if __name__ == '__main__':
    main()
