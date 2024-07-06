import torchvision
import os
import torchvision
import einx
import torch
from .data import MASK_ID, get_dataloader

from .imagenet_labels import LABELS


def unpatch(patches, patch_size=14, image_channels=3):
    return einx.rearrange(
        "... S (C PH PW) -> ... S C PH PW",
        patches,
        C=image_channels,
        PH=patch_size,
        PW=patch_size,
    )


def unnorm(im):
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    im = einx.multiply("... c h w, c -> ... c h w", im, std)
    im = einx.add("... c h w, c -> ... c h w", im, mean)
    return im


def draw_seq_(sequence, image, patch_size=14):
    for patch, pi, pj in zip(
        sequence["patches"], sequence["height_ids"], sequence["width_ids"]
    ):
        i = pi * patch_size
        j = pj * patch_size
        image[:, i : i + patch_size, j : j + patch_size] = patch


def draw_zig_zag_(image, stride=14):
    h, w = image.shape[-2:]
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mask = ((i - j) % stride == 0) | ((i + j) % stride == 0)
    mask_idx = mask.nonzero()
    color = torch.tensor([0.9, 0.2, 0.2])
    einx.set_at("... c [h w], i [2], c -> ... c h w", image, mask_idx, color)


def make_viz(context, preds, patch_size=14):
    context["patches"] = unnorm(unpatch(context["patches"]))
    for pred in preds:
        pred["patches"] = unnorm(unpatch(pred["patches"]))

    nph = max(
        context["height_ids"].max().item(),
        max(pred["height_ids"].max().item() for pred in preds),
    )
    npw = max(
        context["width_ids"].max().item(),
        max(pred["width_ids"].max().item() for pred in preds),
    )

    h = nph * patch_size + patch_size
    w = npw * patch_size + patch_size

    image = torch.zeros(3, h, w)

    draw_seq_(context, image, patch_size)

    for pred in preds:
        draw_zig_zag_(pred["patches"], patch_size)
        draw_seq_(pred, image, patch_size)

    image = (image * 255).to(torch.uint8)

    torchvision.transforms.Resize((int(h * 1.5), int(w * 1.5)))(image)

    return image


def viz(conf):
    torch.manual_seed(42)
    sequence_length_context = conf.dataset_train.sequence_length_context
    dataloader = get_dataloader(
        batch_size=16, packer_batch_size=8, **conf.dataset_train
    )

    num_batches = 8

    for i, batch in enumerate(dataloader):
        patches, metadata_batch = batch

        for j, (sequence, metadata) in enumerate(zip(patches.iloc, metadata_batch)):
            context_seq = sequence.iloc[:sequence_length_context]
            pred_seq = sequence.iloc[sequence_length_context:]

            for id in torch.unique(sequence["sequence_ids"]):
                id = id.item()
                if id == MASK_ID:
                    continue

                pred_image_seq = pred_seq.iloc[pred_seq["sequence_ids"] == id]
                context_image_seq = context_seq.iloc[context_seq["sequence_ids"] == id]

                prediction_block_masks = pred_image_seq[
                    "prediction_block_masks"
                ].unbind(-1)

                label = metadata[id]["label"]
                label_name = LABELS[label]

                preds = []
                for prediction_block_mask in prediction_block_masks:
                    pred = pred_image_seq.iloc[prediction_block_mask]
                    preds.append(pred)

                image = make_viz(context_image_seq, preds, conf.patch_size)

                file = f"out/viz/batch{i:03}-seq{j:03}-id{id}-{label_name}.jpg"
                os.makedirs("out/", exist_ok=True)
                os.makedirs("out/viz/", exist_ok=True)
                torchvision.io.write_jpeg(image, file, 100)
                print("wrote", file)

        if i >= num_batches:
            break
