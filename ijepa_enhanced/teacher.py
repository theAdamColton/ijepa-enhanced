from torch import nn
from .ema import EMA


class Teacher(nn.Module):
    def __init__(self, vit, lfq, beta=0.996):
        super().__init__()
        self.vit = EMA(vit, beta=beta)
        self.lfq = EMA(lfq, beta=beta)

    def forward(self, x, attention_mask, position_indices):
        x = self.vit(x, attention_mask, position_indices)
        lfq_result = self.lfq(x, return_dict=True, return_indices=True)
        return lfq_result["indices"], lfq_result["hidden_states"]

    def update(self):
        self.vit.update()
        self.lfq.update()
