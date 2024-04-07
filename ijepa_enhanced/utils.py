import numpy as np
import torch
import torchvision
from torchvision.io import ImageReadMode
import matplotlib.pyplot as plt


def imread(path):
    return torchvision.io.read_image(path, ImageReadMode.RGB) / 255


def imshow(image: torch.Tensor, ax=None):
    image = image.detach().cpu().moveaxis(0, -1).numpy()
    if ax is None:
        plt.imshow(image)
        plt.show()
    else:
        ax.show(image)
