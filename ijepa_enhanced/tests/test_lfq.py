import unittest
import torch
import einx
from torch import nn
from torch.nn import functional as F

from ..lfq import LFQ, masked_mean, entropy_loss, calculate_perplexity
from ..utils import imread, rand_log_uniform


class TestLFQ(unittest.TestCase):
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
        """
        torch.manual_seed(42)
        for _ in range(100):
            d = dict(
                learning_rate=rand_log_uniform(1e2, 1e-5),
                entropy_weight=rand_log_uniform(1e2, 1e-5),
                commit_weight=rand_log_uniform(1e2, 1e-5),
                entropy_sample_minimization_weight=rand_log_uniform(1, 0),
                entropy_batch_maximization_weight=rand_log_uniform(1, 0),
            )

            loss, perplexity, xhat = do_train(**d)

            print(
                {k: f"{v:.5f}" for k, v in d.items()},
                f"loss {loss.item():.5f} perplexity {perplexity.item():.5f}",
            )


def do_train(
    patch_size=16,
    c=3,
    dim=256,
    codebook_size=256,
    image_file="./images/plume-512x512.jpg",
    learning_rate=1e-3,
    entropy_weight=1e2,
    entropy_sample_minimization_weight=0.0,
    entropy_batch_maximization_weight=1.0,
    commit_weight=1e-9,
    iterations=50,
):
    image = imread(image_file)
    patches = einx.rearrange(
        "c (nh ph) (nw pw) -> nh nw (ph pw c)", image, ph=patch_size, pw=patch_size, c=c
    )
    inproj = nn.Sequential(
        nn.Linear(patch_size * patch_size * 3, dim),
        nn.GELU(),
    )
    lfq = LFQ(
        dim,
        codebook_size,
        sample_minimization_weight=entropy_sample_minimization_weight,
        batch_maximization_weight=entropy_batch_maximization_weight,
    )
    outproj = nn.Sequential(
        nn.GELU(),
        nn.Linear(dim, patch_size * patch_size * c),
    )
    optim = torch.optim.SGD(
        list(inproj.parameters()) + list(lfq.parameters()) + list(outproj.parameters()),
        lr=learning_rate,
    )
    for i in range(iterations):
        z = inproj(patches)
        z, indices, entropy_loss, commit_loss = lfq(
            z, return_losses=True, return_indices=True
        )
        xhat = outproj(z)
        loss = F.mse_loss(patches, xhat)
        loss = loss + entropy_loss * entropy_weight + commit_loss * commit_weight
        loss.backward()
        optim.step()
        optim.zero_grad()

    perplexity = calculate_perplexity(indices, codebook_size)
    loss = F.mse_loss(patches, xhat)
    xhat = einx.rearrange(
        "nh nw (ph pw c) -> c (nh ph) (nw pw)", xhat, ph=patch_size, pw=patch_size, c=c
    )

    return loss, perplexity, xhat
