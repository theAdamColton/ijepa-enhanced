from collections.abc import Iterable
import torch


class TensorSet:
    """
    A small wrapper allowing manipulation of a set of columns,
    each column with the same sequence length (rows)

    Columns do not have to have the same dype or device

    Each column is of shape (B, S, ...) if is_batched
     otherwise shape (S, ...)
     All columns must have matching batch and sequence dims
    """

    def __init__(self, columns: Iterable[torch.Tensor], is_batched=False):
        columns = list(columns)
        self.is_batched = is_batched
        self.columns = columns

    @property
    def num_rows(self):
        if self.columns:
            if self.is_batched:
                return self.columns[0].shape[1]
            else:
                return self.columns[0].shape[0]
        return 0

    @property
    def num_columns(self):
        return len(self.columns)

    def __getitem__(self, key):
        return TensorSet(c[key] for c in self.columns)

    @staticmethod
    def cat(tensorsets):
        num_columns = tensorsets[0].num_columns
        c = []
        axis = 1 if tensorsets[0].is_batched else 0
        for j in range(num_columns):
            c.append(torch.cat([ts.columns[j] for ts in tensorsets], axis))

        return TensorSet(c)

    @staticmethod
    def stack(tensorsets):
        num_columns = tensorsets[0].num_columns
        c = []
        for j in range(num_columns):
            c.append(torch.stack([ts.columns[j] for ts in tensorsets]))

        return TensorSet(c, is_batched=True)

    def pad(self, amount: int, value=-1):
        assert amount >= 0
        padding = []
        for c in self.columns:
            _, *rest_shape = c.shape
            pad_col = torch.full(
                (amount, *rest_shape),
                value,
                dtype=c.dtype,
                device=c.device,
            )
            padding.append(pad_col)
        padding = TensorSet(padding)
        return TensorSet.cat([self, padding])
