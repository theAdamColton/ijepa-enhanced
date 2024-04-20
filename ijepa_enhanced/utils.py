import numpy as np
import torch
import torchvision
import math
import torchvision
from torchvision.io import ImageReadMode
import matplotlib.pyplot as plt


def imread(path):
    return torchvision.io.read_image(path, ImageReadMode.RGB) / 255


def imsave(image, path, norm=False):
    image = image.detach().cpu()
    if image.is_floating_point():
        if norm:
            image = (image - image.min()) / (image.max() - image.min())
        image = image.clamp(0, 1) * 255
        image = image.to(torch.uint8)

    torchvision.io.write_jpeg(image, filename=path, quality=90)
    print("wrote ", path)


def imshow(image: torch.Tensor, ax=None):
    image = image.detach().cpu().moveaxis(0, -1).numpy()
    if ax is None:
        plt.imshow(image)
        plt.show()
    else:
        ax.show(image)


def rand_uniform(a, b):
    u = torch.rand((1,)).item()
    return u * (b - a) + a


def rand_log_uniform(a, b):
    if b < a:
        tmp = b
        b = a
        a = tmp
    if a == 0.0:
        a = 1e-15
    return math.e ** rand_uniform(math.log(a), math.log(b))


def print_num_parameters(model: torch.nn.Module):
    n = sum(p.numel() for p in model.parameters())
    print(f"{n/1_000_000} million parameters")
