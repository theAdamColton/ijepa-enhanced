import random
import torch
import unittest

import scipy
from torch.nn import CrossEntropyLoss
from .. import patchnpack
from .utils import imread, imshow
import os


def is_basically_equal(values, value):
    p = scipy.stats.ttest_1samp(values, value).pvalue
    return p < 0.05


class TestPatchNPack(unittest.TestCase):
    def load_test_images(self, patch_size):
        images = ["images/" + f for f in os.listdir("./images/")]
        images = [imread(image) for image in images]
        images = [patchnpack.CropToMultipleOf(patch_size)(image) for image in images]
        return images

    def test_patch_unpack(self):
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
        packer = patchnpack.PatchNPacker(patch_size, 32, 1)
        images = self.load_test_images(patch_size)
        [packer.append(image) for image in images]
        while packer.can_pop_batch():
            batch = packer.pop_batch()
            patches, positions, image_ids = batch.columns
            patches = patches[0]
            positions = positions[0]
            image_ids = image_ids[0]

            images = patchnpack.unpack(patches, positions, image_ids, patch_size, 3)

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
        for _ in range(10):
            sequence_length = random.randint(3, 373)
            patch_size = random.randint(1, 23)
            batch_size = random.randint(1, 10)
            packer = patchnpack.PatchNPacker(
                patch_size=patch_size,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            n_images = random.randint(3, 50)
            images = []
            c = 3
            for _ in range(n_images):
                h = random.randint(patch_size, 333)
                h = round((h // patch_size) * patch_size)
                w = random.randint(patch_size, 333)
                w = round((w // patch_size) * patch_size)
                image = torch.empty(c, h, w)
                images.append(image)
                packer.append(image)
            while packer.can_pop_batch():
                batch = packer.pop_batch()
                patches, positions, image_ids = batch.columns
                self.assertEqual(sequence_length, patches.shape[1])
                self.assertEqual(sequence_length, positions.shape[1])
                self.assertEqual(sequence_length, image_ids.shape[1])
                for patch_seq, pos_seq, image_ids_seq in zip(
                    patches, positions, image_ids
                ):
                    images = patchnpack.unpack(
                        patch_seq, pos_seq, image_ids_seq, patch_size, c
                    )
                self.assertEqual(sequence_length, batch.num_rows)

    def test_context_target_images(self):
        rng = torch.manual_seed(42)
        random.seed(42)
        patch_size = 16
        sequence_length_context = 256
        sequence_length_target = 512
        ctpacker = patchnpack.ContextTargetPatchNPacker(
            sequence_length_context, sequence_length_target, patch_size, 2, rng=rng
        )
        images = self.load_test_images(patch_size)
        image_ids = list(range(len(images)))
        [ctpacker.append(image, id) for image, id in zip(images, image_ids)]
        batches = []
        while ctpacker.can_pop_batch():
            batches.append(ctpacker.pop_batch())
        rec_images = []
        for batch in batches:
            batch_context, batch_target = batch
            patches, positions, ids, *_ = batch_context.columns
            rec_images.extend(patchnpack.unpack(patches, positions, ids, patch_size, 3))
        # [imshow(im) for im in rec_images]

    def test_sample_rect_mask(self):
        random.seed(100)
        for _ in range(1000):
            h = random.randint(5, 100)
            w = random.randint(5, 100)
            scale = patchnpack.random_uniform(0.1, 0.5)
            mask = patchnpack.sample_rect_mask(h, w, scale, scale, 0.5, 1.5)
            # imshow(mask)
            # print(scale, (mask * 1.0).mean())

    def test_rest_mask_is_centered(self):
        """
        the mean of the center of the random masked pixels should be be 50% of the width and 50% of the height.
        """
        random.seed(100)
        hs, ws = [], []
        for _ in range(10000):
            h = random.randint(5, 200)
            w = random.randint(5, 200)
            mask = patchnpack.sample_rect_mask(h, w, 0.0, 1.0, 0.5, 1.5)
            ch, cw = (mask.nonzero() * 1.0).mean(0)
            ch = ch / h
            cw = cw / w
            hs.append(ch)
            ws.append(cw)
        hs = torch.stack(hs)
        ws = torch.stack(ws)

        self.assertTrue(is_basically_equal(hs, 0.5))
        self.assertTrue(is_basically_equal(ws, 0.5))
