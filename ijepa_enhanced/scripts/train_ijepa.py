from typing import Optional
import torch
import hydra

from ..vit import ViT
from ..lfq import LFQ
from ..patchnpack import ContextTargetPatchNPacker
from ..utils import print_num_parameters


def train_step(vit, predictor, optimizer):
    pass


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(cfg):
    print(cfg)
    vit = ViT(**cfg.vit)
    print("vit: ", end="")
    print_num_parameters(vit)


if __name__ == "__main__":
    main()
