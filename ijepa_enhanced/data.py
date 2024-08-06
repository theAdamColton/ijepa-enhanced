import webdataset as wds
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision
import cabbage_patch
import tensorset as ts

from .mask import MakeContextPredMaskArgs, Optional, make_context_prediction_masks

MASK_ID = -100

pad_value_dict = {
    "patches": 0,
    "sequence_ids": -100,
    "height_ids": 0,
    "width_ids": 0,
    "prediction_block_masks": 0,
}


def rand_a_b(a, b):
    u = torch.rand((1,))
    return (b - a) * u + a


class FilterMinRes(nn.Module):
    def __init__(self, min_res: int = 16):
        super().__init__()
        self.min_res = min_res

    def forward(self, row):
        pixel_values = row["pixel_values"]
        *_, h, w = pixel_values.shape
        return h >= self.min_res and w >= self.min_res


def get_resized_h_w(
    h,
    w,
    min_side_res: int = 64,
    max_side_res: int = 512,
    patch_size: int = 14,
    max_abs_res: int = 1024,
    max_h=None,
    max_w=None,
):
    side_res = int(rand_a_b(min_side_res, max_side_res))
    ar = h / w

    nw = ((side_res**2) / ar) ** 0.5
    nh = ar * nw

    if max_w:
        nw = min(nw, max_w)
    if max_h:
        nh = min(nh, max_h)

    nph = int(max(nh // patch_size, 1))
    npw = int(max(nw // patch_size, 1))

    nh = nph * patch_size
    nw = npw * patch_size

    nh = min(max_abs_res, nh)
    nw = min(max_abs_res, nw)
    return nh, nw


class RandomCrop(nn.Module):
    def __init__(
        self, min_side_res: int, max_side_res: int, patch_size: int, max_abs_res: int
    ):
        super().__init__()
        self.min_side_res = min_side_res
        self.max_side_res = max_side_res
        self.max_abs_res = max_abs_res
        self.patch_size = patch_size

    def forward(self, pixel_values):
        *_, h, w = pixel_values.shape
        nh, nw = get_resized_h_w(
            h,
            w,
            self.min_side_res,
            self.max_side_res,
            self.patch_size,
            self.max_abs_res,
            max_h=h,
            max_w=w,
        )
        crop_fn = torchvision.transforms.RandomCrop((nh, nw))
        return crop_fn(pixel_values)


class RandomResizedCrop(nn.Module):
    def __init__(
        self,
        min_side_res: int,
        max_side_res: int,
        patch_size: int,
        max_abs_res: int,
        crop_scale=(0.3, 1.0),
        ratio=(1 / 3, 3 / 1),
    ):
        super().__init__()
        self.min_side_res = min_side_res
        self.max_side_res = max_side_res
        self.max_abs_res = max_abs_res
        self.patch_size = patch_size
        self.crop_scale = crop_scale
        self.ratio = ratio

    def forward(self, pixel_values):
        *_, h, w = pixel_values.shape
        nh, nw = get_resized_h_w(
            h,
            w,
            self.min_side_res,
            self.max_side_res,
            self.patch_size,
            self.max_abs_res,
        )
        crop_fn = torchvision.transforms.RandomResizedCrop(
            (nh, nw), scale=self.crop_scale, ratio=self.ratio
        )
        return crop_fn(pixel_values)


class RandomResize(nn.Module):
    def __init__(
        self, min_side_res: int, max_side_res: int, patch_size: int, max_abs_res: int
    ):
        super().__init__()
        self.min_side_res = min_side_res
        self.max_side_res = max_side_res
        self.max_abs_res = max_abs_res
        self.patch_size = patch_size

    def forward(self, pixel_values):
        *_, h, w = pixel_values.shape
        nh, nw = get_resized_h_w(
            h,
            w,
            self.min_side_res,
            self.max_side_res,
            self.patch_size,
            self.max_abs_res,
        )
        crop_fn = torchvision.transforms.Resize((nh, nw), antialias=True)
        return crop_fn(pixel_values)


class ContextTargetMaskRow(nn.Module):
    def __init__(self, args: MakeContextPredMaskArgs):
        super().__init__()
        self.args = args

    def forward(self, row):
        patches = row.pop("patches")
        pred_masks = make_context_prediction_masks(
            patches["height_ids"], patches["width_ids"], self.args
        )
        patches["prediction_block_masks"] = pred_masks

        pred_masks_any = pred_masks.any(dim=-1)
        prediction_patches = patches.iloc[pred_masks_any]
        context_patches = patches.iloc[~pred_masks_any]

        row["prediction_patches"] = prediction_patches
        row["context_patches"] = context_patches
        return row


def has_patches(row):
    return row["patches"].size(0) > 0


def has_context_patches(row):
    return row["context_patches"].size(0) > 0


def has_prediction_patches(row):
    prediction_patches = row["prediction_patches"]
    if prediction_patches.size(0) <= 0:
        return False
    prediction_block_masks = prediction_patches["prediction_block_masks"]
    if not prediction_block_masks.sum(0).all():
        return False
    return True


def combine_context_targets(row):
    context_patches = row.pop("context_patches")
    prediction_patches = row.pop("prediction_patches")
    row["patches"] = ts.cat(
        [context_patches, prediction_patches],
        1,
    )
    return row


class TokenDropper(nn.Module):
    """
    Drops tokens randomly
    Each token has a `drop_chance` of being dropped, unless `max_sequence_length` is specified.
    In the case that the sequence length is larger than the `max_sequence_length`,
    tokens will be randomly dropped until the sequence length is equal to `max_sequence_length`.
    """

    def __init__(
        self,
        drop_chance=0.25,
        max_sequence_length: Optional[int] = None,
        torch_rng=None,
    ):
        super().__init__()
        self.drop_chance = drop_chance
        self.max_sequence_length = max_sequence_length
        self.torch_rng = torch_rng

    def forward(self, sequence: ts.TensorSet):
        sequence_length = sequence.size(0)
        if self.drop_chance <= 0.0 and sequence_length <= self.max_sequence_length:
            return sequence

        sample = torch.rand(
            size=(sequence_length,),
            device=sequence.all_columns[0].device,
            generator=self.torch_rng,
        )
        mask = sample > self.drop_chance

        if (
            self.max_sequence_length is not None
            and sequence_length > self.max_sequence_length
        ):
            quantile = sample[mask].quantile(
                1 - self.max_sequence_length / sequence_length
            )
            mask = mask & (sample > quantile)

        sequence = sequence.iloc[mask]
        return sequence


def get_dataloader(
    path,
    min_side_res: int,
    max_side_res: int,
    max_abs_res: int,
    patch_size: int,
    num_samples: Optional[int] = None,
    num_workers: int = 4,
    batch_size: int = 512,
    label_column_name: Optional[str] = None,
    sequence_length_context: int = 128,
    sequence_length_prediction: int = 128,
    drop_chance: float = 0.4,
    packer_batch_size: int = 64,
    do_shuffle: bool = True,
    resize_mode: str = "crop",  # or "resize"
    make_context_pred_mask_args: Optional[
        MakeContextPredMaskArgs
    ] = MakeContextPredMaskArgs(),
):

    assert min_side_res % patch_size == 0
    assert max_side_res % patch_size == 0
    assert max_abs_res % patch_size == 0
    assert batch_size % packer_batch_size == 0

    dataset = (
        cabbage_patch.CabbageDataset(
            path,
            detshuffle=True,
            shardshuffle=True,
            seed=42,
            nodesplitter=wds.split_by_node,
        )
        .decode("torchrgb", handler=wds.handlers.warn_and_continue)
        .rename(pixel_values="jpg")
        .select(FilterMinRes(patch_size))
    )
    if label_column_name:
        dataset = dataset.rename(label="cls")

    if resize_mode == "crop":
        dataset = dataset.map_dict(
            pixel_values=RandomCrop(min_side_res, max_side_res, patch_size, max_abs_res)
        )
    elif resize_mode == "resize":
        dataset = dataset.map_dict(
            pixel_values=RandomResize(
                min_side_res, max_side_res, patch_size, max_abs_res
            )
        )
    elif resize_mode == "random_resized_crop":
        dataset = dataset.map_dict(
            pixel_values=RandomResizedCrop(
                min_side_res, max_side_res, patch_size, max_abs_res
            )
        )
    else:
        raise ValueError(resize_mode)

    max_sequence_length_patches = sequence_length_context
    if make_context_pred_mask_args is not None:
        max_sequence_length_patches += sequence_length_prediction

    dataset = (
        dataset.map_dict(
            pixel_values=torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        )
        .map(cabbage_patch.PatchImageRow(patch_size))
        .map_dict(patches=TokenDropper(drop_chance, max_sequence_length_patches))
        .select(has_patches)
    )

    if num_samples is not None:
        dataset = dataset.with_length(num_samples)
    if do_shuffle:
        dataset = dataset.shuffle(3000)
    if make_context_pred_mask_args is not None:
        dataset = (
            dataset.map(ContextTargetMaskRow(make_context_pred_mask_args))
            .map_dict(
                context_patches=TokenDropper(0.0, sequence_length_context),
                prediction_patches=TokenDropper(0.0, sequence_length_prediction),
            )
            .select(has_context_patches)
            .select(has_prediction_patches)
            .rename(x_patches="context_patches", y_patches="prediction_patches")
            .packed_x_y(
                sequence_length_x=sequence_length_context,
                sequence_length_y=sequence_length_prediction,
                batch_size=packer_batch_size,
                pad_value_dict=pad_value_dict,
            )
            .rename(context_patches="x_patches", prediction_patches="y_patches")
            .map(combine_context_targets)
        )
    else:
        dataset = dataset.packed(
            sequence_length=sequence_length_context,
            batch_size=packer_batch_size,
            pad_value_dict=pad_value_dict,
        )

    if do_shuffle:
        dataset = dataset.shuffle(16)

    dataset = dataset.to_tuple("patches", "metadata").batched(
        batch_size // packer_batch_size, partial=False
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)

    return dataloader
