import unittest
import torch

from ..lfq import LFQ, masked_mean, entropy_loss


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

    def test_encode(self):
        torch.manual_seed(42)
        lfq = LFQ(512, 256)
        x = torch.randn(7, 19, 512)
        lfq.encode(x)

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
