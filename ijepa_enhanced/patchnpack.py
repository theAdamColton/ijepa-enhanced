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
        "... S (C PH PW) -> ... S C PH PW",
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


def get_sample_mask(s: int, max_length: int, rng=None):
    u = torch.rand(s, generator=rng)
    assert s > max_length
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


MASK_IMAGE_ID = 0


def make_tensorset_sequence(
    sequence: List[TensorSet], sequence_length, rng=None
) -> TensorSet:
    sequence = TensorSet.cat(sequence)

    # if the sequence length overflows, randomly drops items in the sequence
    needs_drop = sequence.num_rows > sequence_length
    if needs_drop:
        mask = get_sample_mask(sequence.num_rows, sequence_length, rng)
        sequence = sequence[mask]

    # if the sequence length is too short, pads
    pad_amt = sequence_length - sequence.num_rows
    needs_pad = pad_amt > 0
    if needs_pad:
        sequence = sequence.pad(pad_amt, MASK_IMAGE_ID)

    return sequence


def get_attention_mask(batched_image_ids: torch.LongTensor):
    """
    batched_image_ids: A batch of image ids, shape: (B S)
        where common elements in a batch are identified by identical ids

    returns an attention mask of shape B S S
    """

    attention_mask = einx.rearrange(
        "b i -> b i 1", batched_image_ids
    ) == einx.rearrange("b j -> b 1 j", batched_image_ids)
    attention_mask = attention_mask & (attention_mask != MASK_IMAGE_ID)
    return attention_mask


class PatchNPacker:
    def __init__(self, patch_size, sequence_length, batch_size, rng=None):
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.unpacked_sequences: List[TensorSet] = []
        self.packed_sequences: List[TensorSet] = []
        self.__id = 1
        self.rng = rng
        self._did_flush = False

    def append_image(self, image, id=None):
        assert (
            id != MASK_IMAGE_ID
        ), f"{id} cannot be the same as the mask image id {MASK_IMAGE_ID}"
        patches, positions, image_ids = self.patch(image, id)
        self.__id += 1

        sequence = TensorSet([patches, positions, image_ids])

        self.append_sequence(sequence)

    def patch(self, image, id=None):
        if id == None:
            id = self.__id
        assert (
            id != MASK_IMAGE_ID
        ), f"{id} cannot be the same as the mask image id {MASK_IMAGE_ID}"
        patches, positions = patch(image, self.patch_size)
        s = len(patches)
        image_ids = torch.full((s,), id, dtype=torch.long, device=patches.device)

        return patches, positions, image_ids

    def _flush_sequence(self):
        if len(self.unpacked_sequences) == 0:
            return
        packed_sequences = make_tensorset_sequence(
            self.unpacked_sequences, self.sequence_length
        )
        self.packed_sequences.append(packed_sequences)
        self._did_flush = True

    def append_sequence(self, sequence):
        self._did_flush = False
        s = sum(ts.num_rows for ts in self.unpacked_sequences)

        if s + sequence.num_rows > self.sequence_length and s > 0:
            self._flush_sequence()
            self.unpacked_sequences = [sequence]
        else:
            self.unpacked_sequences.append(sequence)

    def can_pop_batch(self):
        return len(self.packed_sequences) >= self.batch_size

    def pop_batch(self):
        """
        returns a TensorSet, which contains the columns (in this order)
            patches, positions, image_ids
        """
        if not self.can_pop_batch():
            return None
        batch, self.packed_sequences = (
            self.packed_sequences[: self.batch_size],
            self.packed_sequences[self.batch_size :],
        )
        batch = TensorSet.stack(batch)
        # generates attention mask
        attention_mask = get_attention_mask(batch.columns[-1])
        batch.columns.append(attention_mask)

        return batch


class ContextTargetPatchNPacker:
    """
    Allows PackNPatch to be used with context and target patches.
    You can feed a ContextTargetPatchNPacker images and get back
    batches of context patches, and target patches.

    Context/target patching is used in IJepa to construct a input/output signal for predictive modelling.
    """

    def __init__(
        self,
        sequence_length_context,
        sequence_length_target,
        patch_size,
        batch_size,
        num_prediction_targets=4,
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
        self.num_prediction_targets = num_prediction_targets

        self.all_packers = [
            self.patchnpacker_context,
            self.patchnpacker_target,
        ]
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.rng = rng

    def append(self, image, id=None):
        _, h, w = image.shape
        device = image.device
        nph = h // self.patch_size
        npw = w // self.patch_size

        # contains: patches, positions, image ids
        # These are patches for the entire image
        sequence = TensorSet(self.patchnpacker_target.patch(image, id=id))
        assert sequence.num_rows == nph * npw

        # Randomly downsamples the sequence length if it is too long
        if sequence.num_rows > self.sequence_length_target:
            downsample_mask = get_sample_mask(
                sequence.num_rows, self.sequence_length_target, self.rng
            )
            downsample_mask = downsample_mask.to(device)
            sequence = sequence[downsample_mask]
        else:
            downsample_mask = None

        # Samples 4 rectangular target blocks
        target_blocks = []
        for _ in range(self.num_prediction_targets):
            target_block = sample_rect_mask(
                nph,
                npw,
                scale_floor=0.15,
                scale_ceil=0.2,
                aspect_ratio_floor=0.75,
                aspect_ratio_ceil=1.5,
                rng=self.rng,
            ).flatten()

            if downsample_mask is not None:
                target_block = target_block[downsample_mask]

            target_blocks.append(target_block)

        # Samples 1 rectangular context block
        context_block = sample_rect_mask(
            nph, npw, 0.85, 1.0, 1.0, 1.0, rng=self.rng
        ).flatten()

        if downsample_mask is not None:
            context_block = context_block[downsample_mask]

        # the context block might need to be downsampled to fit inside the max context sequence length

        # target_any is a mask that is true if a patch is part of a target block
        # shape: nph * npw
        target_any = sum(target_blocks) > 0

        # context is removed wherever the target mask is true,
        # this is to make it more difficult to predict the target from the context, because they don't overlap
        context_block = context_block & ~target_any

        # Only contains the patches in the context block
        context_sequence = sequence[context_block]

        sequence.columns.extend(target_blocks)

        self.patchnpacker_context.append_sequence(context_sequence)
        self.patchnpacker_target.append_sequence(sequence)

        # if one of the packers flush, they all have to flush
        # this keeps them in sync
        did_flush = any(p._did_flush for p in self.all_packers)
        if did_flush:
            for p in self.all_packers:
                if not p._did_flush:
                    p._flush_sequence()

    def can_pop_batch(self):
        return all(p.can_pop_batch() for p in self.all_packers)

    def pop_batch(self):
        """
        returns a context batch and a target batch as a tuple
        each batch is a TensorSet
        """
        if not self.can_pop_batch():
            return None

        return tuple(p.pop_batch() for p in self.all_packers)
