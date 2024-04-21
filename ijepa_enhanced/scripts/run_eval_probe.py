import hydra
from omegaconf import DictConfig, OmegaConf

from ..vit import ViT
from ..predictor import Predictor
from ..eval import eval_classification_probe


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    vit = ViT(**config.model.vit)
    predictor = Predictor(**config.model.predictor)

    eval_classification_probe(vit, predictor, config.eval)


if __name__ == "__main__":
    main()
