from typing import Optional
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from ..vit import ViT
from ..ema import EMA
from ..predictor import Predictor
from ..lfq import LFQ
from ..patchnpack import ContextTargetPatchNPacker, PatchNPacker
from ..utils import print_num_parameters


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

    patchnpacker = ContextTargetPatchNPacker(**config.train.context_target_patchnpacker)
    print(patchnpacker.batch_size)


if __name__ == "__main__":
    main()
