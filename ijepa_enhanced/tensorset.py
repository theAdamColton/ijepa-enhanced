import torch


class TensorSet:
    """
    A small wrapper allowing manipulation of a set of columns,
    each column with the same sequence length (rows)

    Columns do not have to have the same dype or device
    """

    def __init__(self, *columns):
        if isinstance(columns[0], list):
            columns = columns[0]

        for c in columns:
            assert len(c) == len(columns[0])
        self.columns = columns

    @property
    def num_rows(self):
        if self.columns:
            return len(self.columns[0])
        return 0

    @property
    def num_columns(self):
        return len(self.columns)

    def __getitem__(self, key):
        return TensorSet(c[key] for c in self.columns)

    @staticmethod
    def cat(tensorsets):
        num_columns = tensorsets[0].num_columns
        for ts in tensorsets:
            assert ts.num_columns == num_columns
        c = []
        for j in range(num_columns):
            c.append(torch.cat([ts.columns[j] for ts in tensorsets], 0))

        return TensorSet(c)

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
            padding.append(padding)
        padding = TensorSet(padding)

        return TensorSet.cat([self, padding])
