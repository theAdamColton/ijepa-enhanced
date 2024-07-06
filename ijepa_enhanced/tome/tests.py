import einx
import unittest
import torch

from .tome import TokenMerger


class TestTome(unittest.TestCase):
    def test_constructor(self):
        r = 2
        s = 8
        k = torch.randn(4, s, 1)
        tm = TokenMerger(k, r)
        self.assertEqual(r, tm.dst_idx.size(1))

    def test_merge(self):
        r = 2
        s = 8
        k = torch.randn(4, s, 1)
        tm = TokenMerger(k, r)
        x = torch.randn(4, s, 16)
        tm(x)

    def test_merge_mask(self):
        k = torch.randn(4, 4, 1)
        tm = TokenMerger(k, 1)
        mask = torch.randn(4, 4, 4) > 0
        merged_mask = tm.merge(mask)
        unmerged_mask = tm.unmerge_all(merged_mask)
        self.assertEqual(mask.shape, unmerged_mask.shape)
        self.assertEqual(mask.dtype, unmerged_mask.dtype)

    def test_merge_unmerge_with_ids(self):
        k = torch.randn(1, 32, 1)
        ids = torch.tensor(
            [[0] * 4 + [1] * 4 + [2] * 16 + [3] * 6 + [4] * 2], dtype=torch.long
        )
        tm = TokenMerger(k, 16, sequence_ids=ids)
        merged_ids = tm.merged_ids
        unmerged_ids = tm.unmerge_all(merged_ids)
        self.assertTrue(torch.equal(ids, unmerged_ids))

    def test_merge_layers(self):
        x = torch.tensor(
            [
                [
                    [-0.8892, 1.8355],
                    [-1.1967, -1.2486],
                    [0.5749, 1.9628],
                    [-0.4385, 0.3447],
                    [-0.4215, 0.5183],
                    [0.5829, -0.3063],
                    [-0.4339, -0.0646],
                    [0.9512, 0.6093],
                    [0.2450, 0.2819],
                ]
            ]
        )

        a, b = x[:, ::2], x[:, 1::2]
        scores = einx.dot("b s1 z, b s2 z -> b s1 s2", a, b)

        adm = torch.eye(9)
        adm = einx.rearrange("s1 s2 -> b s1 s2", adm, b=1)

        # merges x a first time
        tm = TokenMerger(x, 3)
        merged_x = tm(x)

        self.assertEqual((1, 6, 2), merged_x.shape)
        rec_x = tm.unmerge_all(merged_x)
        self.assertEqual((1, 9, 2), rec_x.shape)

        # merges x a second time
        tm = TokenMerger(merged_x, 2, adm=tm.adm)
        merged_x = tm(merged_x)
        self.assertEqual((1, 4, 2), merged_x.shape)
        rec_x = tm.unmerge_all(merged_x)
        self.assertEqual((1, 9, 2), rec_x.shape)

    def test_merge_chain(self):
        s = 8
        x = torch.randn(1, s, 10)
        k = torch.randn(1, s, 10)
        r = 2
        tm = TokenMerger(k, r)
        k2 = torch.randn(1, s - r, 10)
        tm2 = tm.chain(k2, r)

        x1 = tm.merge(x)
        x2 = tm2.merge(x1)

        self.assertEqual(x1.size(1), s - r)
        self.assertEqual(x2.size(1), s - r - r)

        x_hat = tm2.unmerge(x2)

        self.assertEqual(x_hat.size(1), s)

    def test_merge_mlerp(self):
        """
        on average, mlerp merging returns tensors with larger magnitudes
        """
        s = 8
        r = 2
        z = 16

        trials = 100
        n = 0
        n_bigger = 0

        for _ in range(trials):
            x = torch.randn(1, s, z)
            k = torch.randn(1, s, 1)
            tm = TokenMerger(k, r)
            x_avg_merged = tm.merge(x, mode="mean")
            x_mlerp_merged = tm.merge(x, mode="mlerp")
            n_bigger += (x_mlerp_merged.abs() > x_avg_merged.abs()).sum()
            n += x_mlerp_merged.shape[0] * x_mlerp_merged.shape[1]

        # p measures how many times the avg_norm was bigger than mlerp_norm
        p = n_bigger / n

        self.assertGreaterEqual(p, 0.95)
