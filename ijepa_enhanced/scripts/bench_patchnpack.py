import random
import os

from .. import patchnpack
from ..utils import imread


if __name__ == "__main__":
    patch_size = 16
    batch_size = 8
    images = ["images/" + f for f in os.listdir("./images/")]
    images = [imread(image) for image in images]
    images = [patchnpack.CropToMultipleOf(patch_size)(image) for image in images]
    ctpacker = patchnpack.ContextTargetPatchNPacker(32, 64, patch_size, batch_size)

    n_images_to_process = 5000
    n_processed = 0

    while n_processed < n_images_to_process:
        ctpacker.append(random.choice(images))
        if ctpacker.can_pop_batch():
            batch = ctpacker.pop_batch()
            n_processed += batch_size
