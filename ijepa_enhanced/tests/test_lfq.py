import numpy as np
import datetime
import unittest
import torch
import torchvision
import einx
from torch import nn
from torch.nn import functional as F
import os

from ..lfq import LFQ, masked_mean, entropy_loss, calculate_perplexity
from ..utils import imread, random_log_uniform


class TestLFQ(unittest.TestCase):
    def setUp(self):
        os.makedirs("./test-files/", exist_ok=True)

    def test_masked_mean(self):
        torch.manual_seed(42)
        for _ in range(5):
            x = torch.randn((1024, 512))
            mask = torch.randn((1024,)) > 0.27
            mean = masked_mean(x, mask).mean()
            self.assertAlmostEqual(mean.item(), x[mask].mean().item())

    def test_entropy_loss(self):
        torch.manual_seed(42)
        for _ in range(5):
            x = torch.randn(128, 64, 1024)
            mask = torch.randn(128) > 0
            masked_loss = entropy_loss(x, mask)
            loss = entropy_loss(x[mask])
            # not exactly the same because of some floating point precision errors when using masked mean
            self.assertAlmostEqual(loss.item(), masked_loss.item(), places=3)

    def test_bits_to_int(self):
        torch.manual_seed(42)
        lfq = LFQ(512, 256)
        for _ in range(5):
            bits = torch.randn(7, 14, 6, 8) > 0
            ints = lfq.bits_to_indices(bits)
            bits_rec = lfq.indices_to_bits(ints)
            self.assertTrue(torch.equal(bits, bits_rec))

    def test_encode_decode(self):
        torch.manual_seed(42)
        for _ in range(10):
            dim = torch.randint(1, 512, (1,)).item()
            log2_codebook_size = torch.randint(1, 9, (1,)).item()
            codebook_size = 2**log2_codebook_size
            lfq = LFQ(dim, codebook_size)
            b = torch.randint(1, 7, (1,)).item()
            s = torch.randint(1, 14, (1,)).item()
            x = torch.randn(b, s, dim)
            with torch.no_grad():
                q, indices = lfq(x, return_indices=True)
                xhat = lfq.decode(indices)
            self.assertTrue(torch.equal(q, xhat))
            self.assertEqual(dim, xhat.shape[-1])
            self.assertSequenceEqual(x.shape, xhat.shape)

    def test_convergence_image_quant(self):
        """
        use downproj, lfq, upproj to compress simple image


        Uses a bunch of random hyperparameters,
        makes sure that on average the loss goes down

        To really make sure this works you want to check the outputted images
        """
        dirname = (
            "./test-files/test-lfq-image-compression/"
            + datetime.datetime.now().ctime()
            + "/"
        )
        os.makedirs("./test-files/test-lfq-image-compression/", exist_ok=True)
        os.makedirs(dirname)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(42)
        loss_diffs = []
        for _ in range(10):
            d = dict(
                learning_rate=random_log_uniform(1e-1, 1e-5),
                entropy_weight=random_log_uniform(1e0, 1e-5),
                commit_weight=random_log_uniform(1e0, 1e-5),
                entropy_sample_minimization_weight=random_log_uniform(100, 1e-4),
                entropy_batch_maximization_weight=random_log_uniform(100, 1e-4),
                temperature=random_log_uniform(100, 0.1),
            )

            result_dict = do_train(device=device, iterations=50, **d)
            xhat = result_dict["xhat"]
            start_loss = result_dict["start_loss"]
            end_loss = result_dict["end_loss"]
            perplexity = result_dict["perplexity"]

            loss_diffs.append(end_loss - start_loss)

            run_string = " ".join([f"{k}:{v:.5f}" for k, v in d.items()])
            run_string += f" loss {end_loss:.5f} perplexity {perplexity:.5f}"

            print(run_string)

            torchvision.io.write_jpeg(
                (xhat * 255).to(torch.uint8),
                dirname + run_string + ".jpg",
                100,
            )

        self.assertLess(np.array(loss_diffs).mean(), 0)


def do_train(
    patch_size=8,
    c=3,
    dim=256,
    codebook_size=256,
    num_codebooks=32,
    image_file="./images/plume-512x512.jpg",
    learning_rate=1e-3,
    entropy_weight=1e2,
    entropy_sample_minimization_weight=0.0,
    entropy_batch_maximization_weight=1.0,
    commit_weight=1e-9,
    iterations=200,
    temperature=0.1,
    device="cpu",
):
    image = imread(image_file).to(device)
    patches = einx.rearrange(
        "c (nh ph) (nw pw) -> nh nw (ph pw c)", image, ph=patch_size, pw=patch_size, c=c
    )
    inproj = nn.Sequential(
        nn.Linear(patch_size * patch_size * 3, dim),
        # nn.GELU(),
    ).to(device)

    lfq = LFQ(
        dim,
        codebook_size,
        num_codebooks=num_codebooks,
        sample_minimization_weight=entropy_sample_minimization_weight,
        batch_maximization_weight=entropy_batch_maximization_weight,
        temperature=temperature,
    ).to(device)

    outproj = nn.Sequential(
        # nn.GELU(),
        nn.Linear(dim, patch_size * patch_size * c),
    ).to(device)
    optim = torch.optim.SGD(
        list(inproj.parameters()) + list(lfq.parameters()) + list(outproj.parameters()),
        lr=learning_rate,
    )

    start_loss = None
    for i in range(iterations):
        z = inproj(patches)
        z, indices, entropy_loss, commit_loss = lfq(
            z, return_losses=True, return_indices=True
        )
        xhat = outproj(z)
        loss = F.mse_loss(patches, xhat)
        if i == 0:
            start_loss = loss.item()

        loss = loss + entropy_loss * entropy_weight + commit_loss * commit_weight
        loss.backward()
        optim.step()
        optim.zero_grad()

    perplexity = calculate_perplexity(indices, codebook_size)
    xhat = xhat.clamp_(0, 1)
    end_loss = F.mse_loss(patches, xhat).item()
    xhat = einx.rearrange(
        "nh nw (ph pw c) -> c (nh ph) (nw pw)", xhat, ph=patch_size, pw=patch_size, c=c
    ).cpu()

    return dict(
        start_loss=start_loss, end_loss=end_loss, perplexity=perplexity, xhat=xhat
    )
