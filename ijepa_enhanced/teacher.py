from torch import nn
from .ema import EMA


class Teacher(nn.Module):
    def __init__(self, vit, lfq, beta=0.996):
        super().__init__()
        self.vit = EMA(vit, beta=beta)
        self.lfq = EMA(lfq, beta=beta)

    def forward(self, *args, **kwargs):
        x = self.vit(*args, **kwargs)
        indices = self.lfq(x, return_dict=True, return_indices=True)["indices"]
        return indices, x

    def update(self):
        self.vit.update()
        self.lfq.update()
