import numpy as np

import data_augs as rad
import kornia
import torch
import os
from os import listdir
from os.path import isfile, join
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_imgs(x, max_display=12):
    grid = (
        make_grid(torch.from_numpy(x[:max_display]), 4).permute(1, 2, 0).cpu().numpy()
    )
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)


def show_stacked_imgs(x, max_display=12):
    fig = plt.figure(figsize=(12, 12))
    stack = 3

    for i in range(1, stack + 1):
        grid = (
            make_grid(torch.from_numpy(x[:max_display, (i - 1) * 3 : i * 3, ...]), 4)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        fig.add_subplot(1, stack, i)
        plt.xticks([])
        plt.yticks([])
        plt.title("frame " + str(i))
        plt.imshow(grid)


def show_channels(img):
    # Split
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Plot
    fig, axs = plt.subplots(2, 2)

    cax_00 = axs[0, 0].imshow(img)
    axs[0, 0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
    axs[0, 0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

    cax_01 = axs[0, 1].imshow(red, cmap="Reds")
    fig.colorbar(cax_01, ax=axs[0, 1])
    axs[0, 1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0, 1].yaxis.set_major_formatter(plt.NullFormatter())

    cax_10 = axs[1, 0].imshow(green, cmap="Greens")
    fig.colorbar(cax_10, ax=axs[1, 0])
    axs[1, 0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1, 0].yaxis.set_major_formatter(plt.NullFormatter())

    cax_11 = axs[1, 1].imshow(blue, cmap="Blues")
    fig.colorbar(cax_11, ax=axs[1, 1])
    axs[1, 1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1, 1].yaxis.set_major_formatter(plt.NullFormatter())

    # Plot histograms
    # fig, axs = plt.subplots(3, sharex=True, sharey=True)

    # axs[0].hist(red.ravel(), bins=10)
    # axs[0].set_title('Red')
    # axs[1].hist(green.ravel(), bins=10)
    # axs[1].set_title('Green')
    # axs[2].hist(blue.ravel(), bins=10)
    # axs[2].set_title('Blue')


if __name__ == "__main__":
    tnsrs_list = list()
    folder = "./mujoco_samples/"
    sample_imgs_files = [
        os.path.join(folder, f) for f in listdir(folder) if isfile(join(folder, f))
    ]

    for fp in sample_imgs_files:
        img = Image.open(fp=fp)
        img_tnsr = transforms.ToTensor()(img)
        img_np = img_tnsr.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        tnsrs_list.append(img_tnsr)

    cat_tnsrs = torch.unsqueeze(torch.cat(tnsrs_list, dim=0), dim=0)

    # show_stacked_imgs(cat_tnsrs.numpy())
    aug_tnsrs = rad.random_convolution(cat_tnsrs)
    img = aug_tnsrs.reshape(-1, 3, 100, 100)[0].detach().numpy()
    # show_stacked_imgs(aug_tnsrs.numpy())
    show_channels(np.transpose(img, (1, 2, 0)))
    plt.savefig("conv.png")
