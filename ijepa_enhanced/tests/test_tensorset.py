import unittest
import torch
import numpy as np
from ..tensorset import TensorSet


class TestTensorSet(unittest.TestCase):
    def test_validate_input_columns(self):
        def assert_value_error(do):
            occurred = False
            try:
                do()
            except ValueError:
                occurred = True
            self.assertTrue(occurred, "value error did not occurr as expected")

        c1 = torch.randn(3, 7, 9)
        c2 = torch.randn(3, 10, 3)

        assert_value_error(lambda: TensorSet([c1, c2], sequence_dim=1))
