import unittest
import torch

from ..lfq import LFQ, masked_mean, entropy_loss


class TestLFQ(unittest.TestCase):
    def test_masked_mean(self):
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn((1024, 512))
            mask = torch.randn((1024,)) > 0.27
            mean = masked_mean(x, mask).mean()
            self.assertAlmostEqual(mean.item(), x[mask].mean().item())

    def test_entropy_loss(self):
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(128, 64, 1024)
            mask = torch.randn(128) > 0
            masked_loss = entropy_loss(x, mask)
            loss = entropy_loss(x[mask])
            self.assertAlmostEqual(loss.item(), masked_loss.item(), places=3)
