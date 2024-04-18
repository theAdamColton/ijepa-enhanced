import torch


def get_optimizer(config, parameters):
    optimizer_cls = getattr(torch.optim, config.name)
    optimizer = optimizer_cls(parameters, **config.args)
    return optimizer
