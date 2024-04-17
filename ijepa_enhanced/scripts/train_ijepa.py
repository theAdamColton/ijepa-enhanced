from typing import Optional
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import accelerate
from torch.utils.data import DataLoader

from ..vit import ViT
from ..ema import EMA
from ..predictor import Predictor
from ..lfq import LFQ
from ..patchnpack import ContextTargetPatchNPacker, PatchNPacker
from ..utils import print_num_parameters
from ..dataset import get_dataset


def train_step(vit, predictor, optimizer):
    pass


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    vit = ViT(**config.vit)
    teacher = EMA(vit, **config.train.ema)
    print("vit: ", end="")
    print_num_parameters(vit)
    predictor = Predictor()
    print("predictor: ", end="")
    print_num_parameters(predictor)

    dataset = get_dataset(**config.data)

    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=config.train.num_workers
    )

    len(dataset)

    patchnpacker = ContextTargetPatchNPacker(**config.train.context_target_patchnpacker)

    accelerator = accelerate.Accelerator()

    optimizer_cls = getattr(torch.optim, config.train.optimizer.name)
    optimizer = optimizer_cls(
        list(vit.parameters()) + list(predictor.parameters()),
        **config.train.optimizer.args
    )


if __name__ == "__main__":
    main()
