"""
This file implements a pure python/pytorch greedy patch n pack algorithm
"""

import random
from typing import List, Optional, Sequence, Sized
from dataclasses import dataclass
from torch import nn
import torch
import einx
from torch._prims_common import TensorSequenceType

from .tensorset import TensorSet


class CropToMultipleOf(nn.Module):
    """
    randomly crops the image to a multiple of size
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, image):
        *_, h, w = image.shape
        assert h >= self.size
        assert w >= self.size
        hf = (h // self.size) * self.size
        wf = (w // self.size) * self.size

        h_diff = h - hf
        w_diff = w - wf

        if h_diff == 0 and w_diff == 0:
            return image

        hs = torch.randint(0, h_diff + 1, (1,))
        ws = torch.randint(0, w_diff + 1, (1,))

        return image[..., hs : hs + hf, ws : ws + wf]


def patch(x: torch.Tensor, patch_size: int):
    """
    x: An image of shape (C H W)

    returns:
        patches: (S Z)
          flattened sequence of patches
        positions (S 2)
          flattened sequence of h,w positions
    """
    c, h, w = x.shape
    assert h % patch_size == 0, h
    assert w % patch_size == 0, w
    nph = h // patch_size
    npw = w // patch_size

    x = torch.reshape(x, (c, nph, patch_size, npw, patch_size))
    x = x.permute(1, 3, 0, 2, 4)
    x = x.reshape(nph * npw, c * patch_size * patch_size)

    device = x.device
    positions = torch.meshgrid(
        torch.arange(nph, device=device),
        torch.arange(npw, device=device),
        indexing="ij",
    )
    positions = torch.stack(positions, -1)
    positions = positions.reshape(nph * npw, 2)

    return x, positions


def unpack(patches, positions, ids, patch_size: int, image_channels: int):
    patches = einx.rearrange(
        "S (C PH PW) -> S C PH PW",
        patches,
        C=image_channels,
        PH=patch_size,
        PW=patch_size,
    )
    images = []
    for id in torch.unique(ids):
        if id == MASK_IMAGE_ID:
            continue
        mask = ids == id
        h = positions[mask][..., 0].max() * patch_size + patch_size
        w = positions[mask][..., 1].max() * patch_size + patch_size
        image = torch.zeros(image_channels, h, w, device=patches.device)
        for pos, patch in zip(positions[mask], patches[mask]):
            i, j = (pos * patch_size).unbind(-1)
            image[
                :,
                i : i + patch_size,
                j : j + patch_size,
            ] = patch

        images.append(image)

    return images


def get_sample_mask(s, max_length: int, rng=None):
    u = torch.rand(s, generator=rng)
    p = max_length / s
    q = u.quantile(p)
    mask = u < q
    return mask


def random_uniform(a=0.0, b=1.0, rng=None):
    u = torch.rand((1,), generator=rng).item()
    return (b - a) * u + a


def sample_rect_mask(
    h: int,
    w: int,
    scale_floor: float = 0.15,
    scale_ceil: float = 0.2,
    aspect_ratio_floor: float = 0.75,
    aspect_ratio_ceil: float = 1.5,
    rng=None,
):
    scale = random_uniform(scale_floor, scale_ceil, rng=rng)
    aspect_ratio = random_uniform(aspect_ratio_floor, aspect_ratio_ceil, rng=rng)
    # block height, block width
    bh = (h * w * scale * aspect_ratio) ** 0.5
    bh = round(bh)
    bh = min(h, bh)
    bw = (h * w * scale / aspect_ratio) ** 0.5
    bw = round(bw)
    bw = min(w, bw)
    # block top left start coords
    bhs = torch.randint(0, h - bh + 1, (1,), generator=rng)
    bws = torch.randint(0, w - bw + 1, (1,), generator=rng)
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[bhs : bhs + bh, bws : bws + bw] = 1
    return mask


MASK_IMAGE_ID = -100


class PatchNPacker:
    def __init__(self, patch_size, sequence_length, batch_size, rng=None):
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.unpacked_sequences: List[TensorSet] = []
        self.packed_sequences: List[TensorSet] = []
        self.__id = 0
        self.rng = rng

    def append(self, image, id=None):
        patches, positions, image_ids = self.patch(image, id)
        self.__id += 1

        sequence = TensorSet([patches, positions, image_ids])

        self._pack(sequence)

    def patch(self, image, id=None):
        if id == None:
            id = self.__id
        patches, positions = patch(image, self.patch_size)
        s = len(patches)
        image_ids = torch.full((s,), id, dtype=torch.long, device=patches.device)

        return patches, positions, image_ids

    def _pack(self, sequence):
        s = sum(ts.num_rows for ts in self.unpacked_sequences)

        if s + sequence.num_rows > self.sequence_length and s > 0:
            packed_sequence = TensorSet.cat(self.unpacked_sequences)
            self.unpacked_sequences = [sequence]

            # if the sequence length overflows, randomly drops items in the sequence
            needs_drop = packed_sequence.num_rows > self.sequence_length
            if needs_drop:
                mask = get_sample_mask(
                    packed_sequence.num_rows, self.sequence_length, self.rng
                )
                packed_sequence = packed_sequence[mask]

            # if the sequence length is too short, masks
            pad_amt = self.sequence_length - packed_sequence.num_rows
            needs_pad = pad_amt > 0
            if needs_pad:
                packed_sequence = packed_sequence.pad(pad_amt, MASK_IMAGE_ID)

            if packed_sequence.num_rows != self.sequence_length:
                import bpdb

                bpdb.set_trace()
            self.packed_sequences.append(packed_sequence)
        else:
            self.unpacked_sequences.append(sequence)

    def can_pop_batch(self):
        return len(self.packed_sequences) >= self.batch_size

    def pop_batch(self):
        if not self.can_pop_batch():
            return None
        batch, self.packed_sequences = (
            self.packed_sequences[: self.batch_size],
            self.packed_sequences[self.batch_size :],
        )
        return TensorSet.stack(batch)


class ContextTargetPatchNPacker:
    """
    Allows PackNPatch to be used with context and target patches.
    You can feed a ContextTargetPatchNPacker images and get back
    batches of context patches, and target patches.

    Context/target patching is used in IJepa to perturb the input signal.
    """

    def __init__(
        self,
        sequence_length_context,
        sequence_length_target,
        patch_size,
        batch_size,
        rng=None,
    ):
        self.sequence_length_context = sequence_length_context
        self.sequence_length_target = sequence_length_target
        self.patchnpacker_context = PatchNPacker(
            patch_size=patch_size,
            sequence_length=sequence_length_context,
            batch_size=batch_size,
        )
        self.patchnpacker_target = PatchNPacker(
            patch_size=patch_size,
            sequence_length=sequence_length_target,
            batch_size=batch_size,
        )
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.rng = rng

    def append(self, image):
        _, h, w = image.shape
        nph, npw = h // self.patch_size

        patches, positions, ids = self.patchnpacker_context.patch(image)
        # sometimes requires a random mask to get the number of target patches to be lower than the max sequence length allowable
        random_mask = get_sample_mask(
            patches.shape[0], self.sequence_length_target, rng=self.rng
        )

        target_blocks = []
        for _ in range(4):
            target_block = sample_rect_mask(
                nph,
                npw,
                scale_floor=0.15,
                scale_ceil=0.2,
                aspect_ratio_floor=0.75,
                aspect_ratio_ceil=1.5,
                rng=self.rng,
            ).flatten()
            target_blocks.append(target_block)

        context_block = sample_rect_mask(
            h, w, 0.85, 1.0, 1.0, 1.0, rng=self.rng
        ).flatten()
        target_any = sum(target_blocks) > 0
        context_block = context_block & ~target_any

        sequence = TensorSet(patches, positions, ids, target_any)
        sequence = sequence[random_mask]
        context_sequence = sequence[context_block]
        self.patchnpacker_context._pack(context_sequence)
        self.patchnpacker_target._pack(sequence)

    def pop_batch(self):
        if (
            not self.patchnpacker_context.can_pop_batch()
            or not self.patchnpacker_target.can_pop_batch()
        ):
            return None

        return (
            self.patchnpacker_context.pop_batch(),
            self.patchnpacker_target.pop_batch(),
        )
