import random
import einx
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import webdataset as wds
import datasets


class CropToMultipleOf(nn.Module):
    """
    randomly crops the image to a multiple of size
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, pixel_values):
        *_, h, w = pixel_values.shape
        assert h >= self.size
        assert w >= self.size
        hf = (h // self.size) * self.size
        wf = (w // self.size) * self.size

        h_diff = h - hf
        w_diff = w - wf

        if h_diff == 0 and w_diff == 0:
            return pixel_values

        hs = torch.randint(0, h_diff + 1, (1,))
        ws = torch.randint(0, w_diff + 1, (1,))

        return pixel_values[..., hs : hs + hf, ws : ws + wf]


def resize_to_max(pixel_values, max_res):
    *_, h, w = pixel_values.shape
    if max(h, w) > max_res:
        aspect_ratio = h / w
        if h > w:
            h = max_res
            w = int(h / aspect_ratio)
        else:
            w = max_res
            h = int(aspect_ratio * w)

        rz = transforms.Resize(min(h, w), antialias=True)
        pixel_values = rz(pixel_values)
    return pixel_values


class ResizeToMax(nn.Module):
    def __init__(self, max_res):
        super().__init__()
        self.max_res = max_res

    def __call__(self, pixel_values):
        pixel_values = resize_to_max(pixel_values, self.max_res)
        return pixel_values


class FilterMinRes(nn.Module):
    def __init__(self, min_res):
        super().__init__()
        self.min_res = min_res

    def __call__(self, row):
        return min(row["pixel_values"].shape[1:]) >= self.min_res


class ToTorchRGB8:
    """
    expects to be fed rows with np rgb8 images
    """

    def __init__(self):
        super().__init__()

    def __call__(self, row):
        pixel_values = row.pop("pixel_values")
        pixel_values = pixel_values.transpose(2, 0, 1)
        pixel_values = torch.from_numpy(pixel_values)
        row["pixel_values"] = pixel_values
        return row


def get_wds_image_dataset(
    path,
    max_res=None,
    min_res=None,
    handler=wds.handlers.reraise_exception,
    image_column_name="jpg",
):
    ds = (
        wds.WebDataset(path)
        .decode("rgb8", handler=handler)
        .rename(pixel_values=image_column_name, handler=handler)
        .map(ToTorchRGB8())
    )

    if min_res:
        ds = ds.select(FilterMinRes(min_res))
    if max_res:
        ds = ds.map_dict(pixel_values=ResizeToMax(max_res), handler=handler)

    return ds


def get_dataset(
    type="hf-image",  # or 'wds-image'
    split="train",
    path="mnist",  # hf dataset path or wds dataset path
    download_url="",  # for debug
    num_classes=None,
    image_column_name="image",
    cls_column_name=None,  # None or string
    max_rows=None,
    crop_to_resolution_multiple_of=None,
    max_res=None,
    seed=42,
):
    """
    returns a iterable, where each row of the iterable will have an accessible 'pixel_values',
     which will be a uint8 3D array: c,h,w torch tensor
    """
    print(f"loading dataset...", path)
    if max_res:
        print(f"  max resolution: {max_res}")
    if crop_to_resolution_multiple_of:
        print(f"  crop to resolution multiple of: {crop_to_resolution_multiple_of}")

    if type == "hf-image":
        ds = datasets.load_dataset(
            path,
            split=split,
        ).rename_column(image_column_name, "pixel_values")
        if cls_column_name:
            ds = ds.rename_column(cls_column_name, "label")
        # ds.set_format("torch", columns=["pixel_values"], output_all_columns=True)

        if max_rows:
            ds = ds.select(range(max_rows))

        def transform(row):
            pixel_values = row.pop("pixel_values")
            pixel_values = np.asarray(pixel_values)
            pixel_values = torch.from_numpy(pixel_values)
            pixel_values = einx.rearrange("... h w c -> ... c h w", pixel_values)

            if max_res is not None:
                pixel_values = resize_to_max(pixel_values, max_res)

            if crop_to_resolution_multiple_of is not None:
                pixel_values = CropToMultipleOf(crop_to_resolution_multiple_of)(
                    pixel_values
                )

            row["pixel_values"] = pixel_values
            return row

        ds.set_transform(transform, columns="pixel_values", output_all_columns=True)
        ds = ds.shuffle(seed)
    elif type == "wds-image":
        ds = get_wds_image_dataset(
            path, image_column_name=image_column_name, max_res=max_res
        )
        ds = ds.with_length(max_rows)
        if crop_to_resolution_multiple_of:
            ds = ds.map_dict(
                pixel_values=CropToMultipleOf(crop_to_resolution_multiple_of)
            )
        ds = ds.shuffle(1000, rng=random.Random(seed))
    else:
        raise ValueError(type)

    return ds
