from torch import nn
from torchvision import transforms
import webdataset as wds
import datasets


def get_dataset(
    type="hf-image",  # or 'wds-image'
    path="mnist",  # hf dataset path or wds dataset path
    download_url="",  # for debug
    image_column_name="image",
    max_rows=None,
):
    """
    returns a iterable, where each row of the iterable will have an accessible 'pixel_values',
     which will be a 0.0 - 1.0 normalized c,h,w torch tensor
    """
    if type == "hf-image":
        ds = get_hf_image_dataset(path, image_column_name)
        if max_rows:
            ds = ds.select(range(max_rows))
    elif type == "wds-image":
        ds = get_wds_image_dataset(path, image_column_name=image_column_name)
        ds = ds.with_length(max_rows)
    else:
        raise ValueError(type)

    return ds


def get_hf_image_dataset(dataset, image_column_name="image"):
    ds = datasets.load_dataset(
        dataset,
        split="train",
    ).rename_column(image_column_name, "pixel_values")
    ds.set_format("torch", columns=["pixel_values"], output_all_columns=True)
    return ds


def resize_to_max(pixel_values, max_res):
    _, h, w = pixel_values.shape
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
        self.max_res = max_res

    def __call__(self, pixel_values):
        pixel_values = resize_to_max(pixel_values, self.max_res)
        return pixel_values


class FilterMinRes:
    def __init__(self, min_res):
        self.min_res = min_res

    def __call__(self, row):
        return min(row["pixel_values"].shape[1:]) >= self.min_res


def get_wds_image_dataset(
    path,
    max_res=None,
    min_res=None,
    handler=wds.handlers.reraise_exception,
    image_column_name="jpg",
):
    ds = (
        wds.WebDataset(path)
        .decode("torchrgb8", handler=handler)
        .rename(pixel_values=image_column_name, handler=handler)
    )

    if min_res:
        ds = ds.select(FilterMinRes(min_res))
    if max_res:
        ds = ds.map_dict(pixel_values=ResizeToMax(max_res), handler=handler)

    return ds
