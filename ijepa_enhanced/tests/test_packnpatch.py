import random
import torch
import unittest

from torch.nn import CrossEntropyLoss
from .. import patchnpack
from .utils import imread, imshow
import os


class TestPatchNPack(unittest.TestCase):
    def test_pack_unpack(self):
        patch_size = 16
        image = imread("images/dog.jpg")
        image = patchnpack.CropToMultipleOf(patch_size)(image)
        patches, positions = patchnpack.patch(image, patch_size)
        ids = torch.full((len(patches),), 0)
        image_hat = patchnpack.unpack(patches, positions, ids, patch_size, 3)[0]
        # imshow(image_hat)
        self.assertTrue(torch.equal(image, image_hat))

    def test_packer_images(self):
        sequence_length = 8
        patch_size = 32
        images = ["images/" + f for f in os.listdir("./images/")]
        images = [imread(image) for image in images]
        images = [patchnpack.CropToMultipleOf(patch_size)(image) for image in images]

    def test_pack_simple(self):
        sequence1 = torch.rand(10)
        sequence2 = torch.rand(14)
        sequence3 = torch.rand(32)
        sequence4 = torch.rand(32)

        batches, unbatched = patchnpack.pack(
            [sequence1, sequence2, sequence3, sequence4], 32
        )

        self.assertTrue(torch.equal(batches[0][: len(sequence1)], sequence1))
        self.assertTrue(
            torch.equal(
                batches[0][len(sequence1) : len(sequence1) + len(sequence2)], sequence2
            )
        )

    def test_patchnpack_pipe(self):
        image = imread("./images/dog.jpg")
        p = 32
        image = patchnpack.CropToMultipleOf(p)(image)
        patches = patchnpack.patch(image, p)

    def test_crop_to_multiple(self):
        random.seed(42)
        for _ in range(20):
            p = random.randint(2, 20)
            h = random.randint(p, 333)
            w = random.randint(p, 333)
            im = torch.empty(h, w)
            im = patchnpack.CropToMultipleOf(p)(im)
            self.assertEqual(im.shape[0] % p, 0)
            self.assertEqual(im.shape[1] % p, 0)

    def test_pack_unpack_random(self):
        rng = torch.manual_seed(42)
        random.seed(42)
        for _ in range(1):
            sequence_length = random.randint(3, 373)
            patch_size = random.randint(1, 23)
            batch_size = random.randint(1, 10)
            packer = patchnpack.PatchNPacker(
                patch_size=patch_size,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            n_images = random.randint(3, 13)
            images = []
            for _ in range(n_images):
                h = random.randint(patch_size, 333)
                h = round((h // patch_size) * patch_size)
                w = random.randint(patch_size, 333)
                w = round((w // patch_size) * patch_size)
                image = torch.empty(3, h, w)
                images.append(image)
                packer.append(image)
            while packer.can_pop_batch():
                batch = packer.pop_batch()
