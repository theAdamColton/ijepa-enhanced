import os
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
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

    dataset = get_dataset(**config.train.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
    )
    dataloader = iter(dataloader)

    for context, target in patchnpacker.make_iter(dataloader):
        batch_size = context.leading_shape[0]
        device = context.all_columns[0].device

        is_prediction_mask = torch.zeros(
            batch_size, context.sequence_length, dtype=torch.bool
        )

        prediction_block_masks = target.named_columns.pop(
            "prediction_block_masks"
        ).unbind(-1)

        target.named_columns["is_prediction_mask"] = torch.ones(
            target.leading_shape, device=device, dtype=torch.bool
        )

        context.named_columns["is_prediction_mask"] = torch.zeros(
            context.leading_shape, device=device, dtype=torch.bool
        )

        for mask_i, prediction_block_mask in enumerate(prediction_block_masks):
            pred = patchnpacker.make_prediction_target_sequence(
                target, context, prediction_block_mask
            )

            # visualizes target patches
            for batch_index in range(pred.leading_shape[0]):
                pred_seq = pred.iloc[batch_index]

                patches = pred_seq["patches"]
                positions = pred_seq["positions"]
                ids = pred_seq["image_ids"]
                is_pad = ids == MASK_IMAGE_ID
                is_prediction_mask = pred_seq["is_prediction_mask"]

                is_not_prediction_mask = ~is_prediction_mask & ~is_pad
                is_prediction_mask = is_prediction_mask & ~is_pad

                ctx_ids = ids[is_not_prediction_mask].unique()
                ctx_ids = set(int(i) for i in ctx_ids)

                tgt_ids = ids[is_prediction_mask].unique()
                tgt_ids = set(int(i) for i in tgt_ids)

                if not ctx_ids == tgt_ids:
                    raise ValueError(
                        (
                            "There needs to exist both ctx and tgt patches for every id",
                            (tgt_ids - ctx_ids, ctx_ids - tgt_ids),
                        )
                    )

                # converts to float and slightly lightens the tgt rectangle
                patches = patches / 255.0
                patches[is_prediction_mask] = patches[is_prediction_mask] + 0.1

                images = unpack(patches, positions, ids, patchnpacker.patch_size, 3)
                ids = ids[ids != MASK_IMAGE_ID].unique()

                for id, image in zip(ids, images):
                    imsave(
                        image,
                        f"viz-output/image-{id.item():04}-mask{mask_i:04}.jpg",
                        norm=True,
                    )


if __name__ == "__main__":
    main()
