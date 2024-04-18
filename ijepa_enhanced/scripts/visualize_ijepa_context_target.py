import hydra
from omegaconf import DictConfig, OmegaConf
from ..patchnpack import ContextTargetPatchNPacker


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
        patchnpacker = ContextTargetPatchNPacker(
        patch_size=config.model.vit.patch_size,
        batch_size=config.train.batch_size,
        sequence_length_context=config.train.sequence_length_context,
        sequence_length_target=config.train.sequence_length_target,
    )




    

if __name__ == "__main__':
    main()
