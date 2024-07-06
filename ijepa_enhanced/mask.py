from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class MakeContextPredMaskArgs:
    num_prediction_targets: int = 4
    pred_scale_floor: float = 0.15
    pred_scale_ceil: float = 0.2
    pred_aspect_ratio_floor: float = 0.75
    pred_aspect_ratio_ceil: float = 1.5


def random_uniform(a=0.0, b=1.0, rng=None):
    u = torch.rand((1,), generator=rng).item()
    return (b - a) * u + a


def sample_rect_size(
    h: int,
    w: int,
    scale_floor: float = 0.15,
    scale_ceil: float = 0.2,
    aspect_ratio_floor: Optional[float] = 0.75,
    aspect_ratio_ceil: Optional[float] = 1.5,
    torch_rng=None,
):
    """
    returns (h, w) of a randomly sized rectangle
    """
    scale = random_uniform(scale_floor, scale_ceil, rng=torch_rng)
    nelement = scale * h * w
    if aspect_ratio_floor and aspect_ratio_ceil:
        aspect_ratio = random_uniform(
            aspect_ratio_floor, aspect_ratio_ceil, rng=torch_rng
        )
    else:
        aspect_ratio = h / w

    # block height, block width
    bh = (nelement * aspect_ratio) ** 0.5
    bh = round(bh)
    bh = min(h, bh)
    bw = (nelement / aspect_ratio) ** 0.5
    bw = round(bw)
    bw = min(w, bw)
    return (bh, bw)


def make_mask(max_h, max_w, height_ids, width_ids, mask_h, mask_w, rng=None):
    top = torch.randint(0, max_h - mask_h + 1, (1,), generator=rng)
    left = torch.randint(0, max_w - mask_w + 1, (1,), generator=rng)
    mask = (
        (height_ids >= top)
        & (height_ids < top + mask_h)
        & (width_ids >= left)
        & (width_ids < left + mask_w)
    )
    return mask


def make_context_prediction_masks(
    height_ids, width_ids, args: MakeContextPredMaskArgs, rng=None
):
    """
    height_ids
        Shape (s,)
    width_ids
        Shape (s,)

    returns:
        pred_masks: Bool tensor of shape (s, num_prediction_targets), contains True where a patch
            is a prediction target
    """
    nph = height_ids.max().item() + 1
    assert nph >= 1
    npw = width_ids.max().item() + 1
    assert npw >= 1

    pred_block_h, pred_block_w = sample_rect_size(
        nph,
        npw,
        args.pred_scale_floor,
        args.pred_scale_ceil,
        args.pred_aspect_ratio_floor,
        args.pred_scale_ceil,
        rng,
    )

    num_teacher_patches = nph * npw

    min_prediction_patches = 1
    max_prediction_patches = max(num_teacher_patches - 4, 1)

    num_trials = 5
    trial = 0

    while trial < num_trials:
        pred_masks = []
        for _ in range(args.num_prediction_targets):
            pred_mask = make_mask(
                nph,
                npw,
                height_ids,
                width_ids,
                pred_block_h,
                pred_block_w,
                rng=rng,
            )
            pred_masks.append(pred_mask)
        pred_masks = torch.stack(pred_masks, -1)

        num_masked = pred_masks.any(-1).sum().item()

        is_good = (
            min_prediction_patches <= num_masked
            and num_masked <= max_prediction_patches
        )

        if is_good:
            return pred_masks

        trial += 1

    print(
        "ran out of attepts to make a mask with between ",
        min_prediction_patches,
        "patches and",
        max_prediction_patches,
        "patches instead got",
        num_masked,
    )
    return pred_masks
