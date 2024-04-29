from copy import deepcopy
import torch
from torch import nn


def inplace_lerp(
    tgt: torch.Tensor,
    src: torch.Tensor,
    weight,
):
    tgt.lerp_(src, weight)


class EMA(nn.Module):
    def __init__(self, model: nn.Module, beta=0.996):
        super().__init__()

        self.beta = beta
        # hack so that the model is not saved in the state dict
        self.model = [model]
        self.ema_model = deepcopy(model).eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        model = self.model[0]
        ema_model = self.ema_model

        for param_model, param_ema_model in zip(
            model.parameters(), ema_model.parameters()
        ):
            if not param_model.is_floating_point():
                continue
            inplace_lerp(param_ema_model, param_model, 1 - self.beta)
        for buffer_model, buffer_ema_model in zip(
            model.parameters(), ema_model.parameters()
        ):
            if not buffer_model.is_floating_point():
                continue
            inplace_lerp(buffer_ema_model, buffer_model, 1 - self.beta)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
