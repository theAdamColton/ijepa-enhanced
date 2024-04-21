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

    for ctx, tgt in patchnpacker.make_iter(dataloader):
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

        tgt_block_masks = [
            tgt.named_columns[f"target_block{i}"]
            for i in range(patchnpacker.num_prediction_targets)
        ]

        is_tgt_mask = torch.ones(batch_size, tgt.sequence_length, dtype=torch.bool)
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
            for pred_seq in pred:
                patches = pred_seq.named_columns["patches"]
                positions = pred_seq.named_columns["positions"]
                ids = pred_seq.named_columns["image_ids"]
                is_tgt_mask = pred_seq.named_columns["is_tgt_mask"]
                is_pad = ids == MASK_IMAGE_ID
                is_not_tgt_mask = ~is_tgt_mask & ~is_pad
                is_tgt_mask = is_tgt_mask & ~is_pad

                ctx_ids = ids[is_not_tgt_mask].unique()
                ctx_ids = set(int(i) for i in ctx_ids)

                tgt_ids = ids[is_tgt_mask].unique()
                tgt_ids = set(int(i) for i in tgt_ids)

                assert (
                    ctx_ids == tgt_ids
                ), "There should be both ctx and tgt patches for every id"

                # converts to float and slightly lightens the tgt rectangle
                patches = patches / 255.0
                patches[is_tgt_mask] = patches[is_tgt_mask] + 0.1

                images = unpack(patches, positions, ids, patchnpacker.patch_size, 3)
                ids = ids[ids != MASK_IMAGE_ID].unique()

                for id, image in zip(ids, images):
                    imsave(
                        image,
                        f"viz-output/image-{id.item():04}-mask{j:04}.jpg",
                        norm=True,
                    )


if __name__ == "__main__":
    main()
