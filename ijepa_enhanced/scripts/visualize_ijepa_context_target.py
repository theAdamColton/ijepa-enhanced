import os
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from ..patchnpack import MASK_IMAGE_ID, ContextTargetPatchNPacker, unpack
from ..dataset import get_dataset
from tensorsequence import TensorSequence
from ..utils import imshow, imsave


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
    torch.manual_seed(config.seed)

    os.makedirs("viz-output/", exist_ok=True)
    patchnpacker = ContextTargetPatchNPacker(
        patch_size=config.model.vit.patch_size,
        batch_size=config.train.batch_size,
        sequence_length_context=config.train.sequence_length_context,
        sequence_length_target=config.train.sequence_length_target,
        sequence_length_prediction=config.train.sequence_length_prediction,
    )

    max_res = config.model.vit.patch_size * min(
        config.model.vit.max_height, config.model.vit.max_width
    )

    dataset = get_dataset(max_res=max_res, **config.train.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.train.num_workers,
    )
    dataloader = iter(dataloader)

    for batch_i, (ctx, tgt) in enumerate(patchnpacker.make_iter(dataloader)):
        batch_size = ctx.all_columns[0].shape[0]
        is_tgt_mask = torch.zeros(batch_size, ctx.sequence_length, dtype=torch.bool)
        ctx = TensorSequence(
            named_columns=dict(
                patches=ctx.named_columns["patches"],
                positions=ctx.named_columns["positions"],
                image_ids=ctx.named_columns["image_ids"],
                is_tgt_mask=is_tgt_mask,
            ),
            sequence_dim=ctx.sequence_dim,
        )

        is_tgt_mask = torch.ones(batch_size, tgt.sequence_length, dtype=torch.bool)
        tgt_block_masks = [
            tgt.named_columns[f"target_block{i}"]
            for i in range(patchnpacker.num_prediction_targets)
        ]

        tgt = TensorSequence(
            named_columns=dict(
                patches=tgt.named_columns["patches"],
                positions=tgt.named_columns["positions"],
                image_ids=tgt.named_columns["image_ids"],
                is_tgt_mask=is_tgt_mask,
            ),
            sequence_dim=tgt.sequence_dim,
        )

        for j, tgt_block_mask in enumerate(tgt_block_masks):
            pred = patchnpacker.make_prediction_target_sequence(
                tgt, ctx, tgt_block_mask
            )

            # visualizes target patches
            for i, pred_seq in enumerate(pred):
                patches = pred_seq.named_columns["patches"]
                positions = pred_seq.named_columns["positions"]
                ids = pred_seq.named_columns["image_ids"]
                is_tgt_mask = pred_seq.named_columns["is_tgt_mask"]
                is_pad = ids == MASK_IMAGE_ID
                is_tgt_mask = is_tgt_mask & ~is_pad

                # converts to f32 and slightly lightens the tgt rectangle
                patches = patches / 255.0
                patches[is_tgt_mask] = patches[is_tgt_mask] + 0.1

                images = unpack(patches, positions, ids, patchnpacker.patch_size, 3)

                for image_i, image in enumerate(images):
                    imsave(
                        image,
                        f"viz-output/batch{batch_i:04}-sequence{i:04}-image{image_i:04}-mask{j:04}.jpg",
                        norm=True,
                    )


if __name__ == "__main__":
    main()
