import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import accelerate
from torch.utils.data import DataLoader

from ijepa_enhanced.tensorset import TensorSet

from ..vit import ViT
from ..ema import EMA
from ..predictor import Predictor
from ..lfq import LFQ, masked_mean
from ..patchnpack import (
    MASK_IMAGE_ID,
    ContextTargetPatchNPacker,
    PatchNPacker,
    get_attention_mask,
)
from ..utils import print_num_parameters
from ..dataset import get_dataset
from ..optimizer import get_optimizer
from ..eval import eval_classification_probe


def train_step(vit, predictor, optimizer):
    pass


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    vit = ViT(**config.model.vit)
    predictor = Predictor(**config.model.predictor)

    eval_classification_probe(vit, predictor, config.eval)


if __name__ == "__main__":
    main()
