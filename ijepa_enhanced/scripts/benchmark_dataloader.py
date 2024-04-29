import time
from tqdm import tqdm
import torch
import hydra
from torch.utils.data import DataLoader


from ..dataset import get_dataset
from ..patchnpack import (
    MASK_IMAGE_ID,
    ContextTargetPatchNPacker,
    get_attention_mask,
)


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config):
    dataset = get_dataset(**config.train.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
    )

    patchnpacker = ContextTargetPatchNPacker(
        **config.train.patchnpack_args,
    )

    total_images_processed = 0
    max_images = 100_000
    progress_bar = tqdm(total=max_images)

    st = time.time()

    for ctx, tgt in patchnpacker.make_iter(dataloader):
        prediction_block_masks = tgt.named_columns.pop("prediction_block_masks").unbind(
            -1
        )

        # here is where the tgt patches would be embedded using the teacher model
        # and the ctx patches would be embedded using the vit

        for prediction_block_mask in prediction_block_masks:
            preds = patchnpacker.pack_prediction_target_sequence(
                tgt, ctx, prediction_block_mask
            )

            attention_mask = get_attention_mask(preds["image_ids"])

            # here is where the tgt predictions would be made using the predictor

        unique_ids = torch.unique(ctx["image_ids"])
        unique_ids = unique_ids[unique_ids != MASK_IMAGE_ID]

        embeddings = []
        labels = []
        for id in unique_ids:
            mask = tgt["image_ids"] == id
            image_sequence = tgt.iloc[mask]
            embedding = image_sequence["patches"].to(torch.float).mean(0)
            label = image_sequence["label"][0]

            embeddings.append(embedding)
            labels.append(label)

        embeddings = torch.stack(embeddings)
        labels = torch.stack(labels)
        # here is where the linear model would be used to predict the labels from the embeddings

        n_images = unique_ids.nelement()
        total_images_processed += n_images
        progress_bar.update(n_images)

        if total_images_processed > max_images:
            print(f"{total_images_processed} in {time.time() - st}")
            return


if __name__ == "__main__":
    main()
