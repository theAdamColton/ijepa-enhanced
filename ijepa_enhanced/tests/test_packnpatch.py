import random
import torch
import unittest

from torch.nn import CrossEntropyLoss
from .. import packnpatch
from .utils import imread, imshow
import os


class TestPacknPatch(unittest.TestCase):
    def test_pack_unpack(self):
        patch_size = 16
        image = imread("images/dog.jpg")
        image = packnpatch.CropToMultipleOf(patch_size)(image)
        patches, positions = packnpatch.patch(image, patch_size)
        ids = torch.full((len(patches),), 0)
        image_hat = packnpatch.unpack(patches, positions, ids, patch_size, 3)[0]
        imshow(image_hat)
        self.assertTrue(torch.equal(image, image_hat))

    def test_packer_images(self):
        sequence_length = 8
        patch_size = 32
        images = ["images/" + f for f in os.listdir("./images/")]
        images = [imread(image) for image in images]
        images = [packnpatch.CropToMultipleOf(patch_size)(image) for image in images]
        packer = packnpatch.Packer(sequence_length)
        for i, image in enumerate(images):
            patches, positions = packnpatch.patch(image, patch_size)
            ids = torch.full((len(patches),), i)
            packer.pack([patches], [positions], [ids])

        batched_patches, batched_positions, batched_ids = packer.flush_batched_tensors()
        for patches, positions, ids in zip(
            batched_patches, batched_positions, batched_ids
        ):
            recimages = packnpatch.unpack(patches, positions, ids, patch_size, 3)
            for image in recimages:
                pass
                # imshow(image)

    def test_pack_simple(self):
        sequence1 = torch.rand(10)
        sequence2 = torch.rand(14)
        sequence3 = torch.rand(32)
        sequence4 = torch.rand(32)

        batches, unbatched = packnpatch.pack(
            [sequence1, sequence2, sequence3, sequence4], 32
        )

        self.assertTrue(torch.equal(batches[0][: len(sequence1)], sequence1))
        self.assertTrue(
            torch.equal(
                batches[0][len(sequence1) : len(sequence1) + len(sequence2)], sequence2
            )
        )

    def test_packnpatch_pipe(self):
        image = imread("./images/dog.jpg")
        p = 32
        image = packnpatch.CropToMultipleOf(p)(image)
        patches = packnpatch.patch(image, p)

    def test_crop_to_multiple(self):
        random.seed(42)
        for _ in range(20):
            p = random.randint(2, 20)
            h = random.randint(p, 333)
            w = random.randint(p, 333)
            im = torch.empty(h, w)
            im = packnpatch.CropToMultipleOf(p)(im)
            self.assertEqual(im.shape[0] % p, 0)
            self.assertEqual(im.shape[1] % p, 0)
