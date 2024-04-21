"""
This file implements a pure python/pytorch greedy patch n pack algorithm
"""

import random
from typing import List, Optional, Sequence, Sized
from dataclasses import dataclass
from torch import nn
import torch
import einx

from tensorsequence import TensorSequence


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
    """
    patches: (B S Z)
    positions: (B S 2)
    ids: (B S)

    Takes a sequence of patches, positions, and image ids
    Images in the sequence are uniquely identified by an id

    Uses the positions to place the patches back in a reconstructed image
    """
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
        image_positions = positions[mask]
        image_heights, image_widths = image_positions.unbind(-1)
        min_h = image_heights.min() * patch_size
        min_w = image_widths.min() * patch_size
        max_h = image_heights.max() * patch_size + patch_size
        max_w = image_widths.max() * patch_size + patch_size
        h = max_h - min_h
        w = max_w - min_w
        image = torch.zeros(
            image_channels, h, w, device=patches.device, dtype=patches.dtype
        )
        for pos, patch in zip(positions[mask], patches[mask]):
            i, j = (pos * patch_size).unbind(-1)
            i = i - min_h
            j = j - min_w
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


MASK_IMAGE_ID = -100


def make_tensorset_sequence(
    sequence: List[TensorSequence], sequence_length, rng=None
) -> TensorSequence:
    sequence = TensorSequence.cat(sequence)

    # if the sequence length overflows, randomly drops items in the sequence
    needs_drop = sequence.sequence_length > sequence_length
    if needs_drop:
        mask = get_sample_mask(sequence.sequence_length, sequence_length, rng)
        sequence = sequence[mask]

    # if the sequence length is too short, pads
    pad_amt = sequence_length - sequence.sequence_length
    needs_pad = pad_amt > 0
    if needs_pad:
        sequence = sequence.pad(pad_amt, MASK_IMAGE_ID)

    return sequence


def get_attention_mask(batched_image_ids: torch.LongTensor) -> torch.BoolTensor:
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


class MakeIterable:
    def make_iter(self, data_iter):
        """
        pass in a data iter which yields rows of data,
            must contain at least "pixel_values"

        returns a generator that yields batches
        """
        while True:
            if not self.can_pop_batch():
                try:
                    row = next(data_iter)
                except StopIteration:
                    return
                image = row["pixel_values"]

                id = row.get("__id__")

                other_ids = []
                label = row.get("label")
                if label is not None:
                    other_ids.append(label)

                self.append_image(image, id=id, label=label)
                continue

            yield self.pop_batch()


class PatchNPacker(MakeIterable):
    def __init__(self, patch_size, sequence_length, batch_size, rng=None):
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.unpacked_sequences: List[TensorSequence] = []
        self.packed_sequences: List[TensorSequence] = []
        self.__id = 1
        self.rng = rng
        self._did_flush = False

    def append_image(
        self,
        image,
        id=None,
        *metadata_ids,
        **named_metadata_ids,
    ):
        """
        Append an image to be packed, along with any metadata ids

        Each additional metadata_id/named_metadata_id if specified has to be an integer
        """
        assert (
            id != MASK_IMAGE_ID
        ), f"{id} cannot be the same as the mask image id {MASK_IMAGE_ID}"

        patches, positions = self.patch(image)

        if id == None:
            id = self.__id
        self.__id += 1

        s = len(patches)

        full = lambda x: torch.full((s,), x, dtype=torch.long, device=patches.device)
        image_ids = full(id)

        metadata_ids = [full(x) for x in metadata_ids]
        named_metadata_ids = {k: full(x) for k, x in named_metadata_ids.items()}

        sequence = TensorSequence(
            [patches, positions, image_ids, *metadata_ids], named_metadata_ids
        )

        self._append_sequence(sequence)

    def patch(self, image):
        patches, positions = patch(image, self.patch_size)
        return patches, positions

    def reset(self):
        self.packed_sequences = []
        self.unpacked_sequences = []
        self._did_flush = False
        self.__id = 1

    def _flush_sequence(self):
        if len(self.unpacked_sequences) == 0:
            return
        packed_sequences = make_tensorset_sequence(
            self.unpacked_sequences, self.sequence_length
        )
        self.packed_sequences.append(packed_sequences)
        self._did_flush = True

    def _append_sequence(self, sequence):
        self._did_flush = False
        s = sum(ts.sequence_length for ts in self.unpacked_sequences)

        if s + sequence.sequence_length > self.sequence_length and s > 0:
            self._flush_sequence()
            self.unpacked_sequences = [sequence]
        else:
            self.unpacked_sequences.append(sequence)

    def can_pop_batch(self):
        return len(self.packed_sequences) >= self.batch_size

    def pop_batch(self):
        """
        returns a TensorSequence, which contains the columns (in this order)
            patches, positions, image_ids
        """
        if not self.can_pop_batch():
            return None
        batch, self.packed_sequences = (
            self.packed_sequences[: self.batch_size],
            self.packed_sequences[self.batch_size :],
        )
        batch = TensorSequence.stack(batch)

        return batch


class ContextTargetPatchNPacker(MakeIterable):
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
        sequence_length_prediction,
        patch_size,
        batch_size,
        num_prediction_targets=4,
        rng=None,
    ):
        self.sequence_length_context = sequence_length_context
        self.sequence_length_target = sequence_length_target
        self.sequence_length_prediction = sequence_length_prediction
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
        self.__id = 0

    def append_image(
        self,
        image,
        id=None,
        **named_metadata_ids,
    ):
        _, h, w = image.shape
        device = image.device
        nph = h // self.patch_size
        npw = w // self.patch_size

        if id is None:
            id = self.__id
            self.__id += 1

        patches, positions = self.patchnpacker_target.patch(image)

        sequence_length = len(patches)

        full = lambda x: torch.full(
            (sequence_length,), x, dtype=torch.long, device=image.device
        )
        image_ids = full(id)

        named_metadata_ids = {k: full(x) for k, x in named_metadata_ids.items()}

        named_columns = {
            "patches": patches,
            "positions": positions,
            "image_ids": image_ids,
            **named_metadata_ids,
        }

        # contains: patches, positions, image ids
        # These are patches for the entire image
        sequence = TensorSequence(
            named_columns=named_columns,
        )

        assert sequence.sequence_length == nph * npw

        # Randomly downsamples the sequence length if it is too long
        if sequence.sequence_length > self.sequence_length_target:
            downsample_mask = get_sample_mask(
                sequence.sequence_length, self.sequence_length_target, self.rng
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

        for i, target_block in enumerate(target_blocks):
            sequence.named_columns[f"target_block{i}"] = target_block

        self.patchnpacker_context._append_sequence(context_sequence)
        self.patchnpacker_target._append_sequence(sequence)

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
        each batch is a TensorSequence
        """
        if not self.can_pop_batch():
            return None

        return tuple(p.pop_batch() for p in self.all_packers)

    def make_prediction_target_sequence(
        self, tgt: TensorSequence, ctx: TensorSequence, tgt_block_mask
    ):
        """
        tgt_block_mask is a boolean mask of shape (B S) which is set to True for sequence
         elements that are prediction targets.

        tgt is a TensorSequence containing patches and other sequence data of the data to be predicted

        ctx is a TensorSequence containing patches and other sequence data of the data to be used as the known independant variable

        Returns a TensorSequence, where each sequence has the elements to be predicted and the context elements concatenated along the sequence dimension.

        The sequence length of the returned TensorSequence is self.sequence_length_prediction + self.sequence_length_context

        TODO could potentially waste less padding by unpacking the ctx_sequence and then repacking it together with the pred_sequence
        """
        device = tgt.all_columns[0].device

        tgt_seq_packed = []
        for tgt_seq_mask, tgt_seq, ctx_seq in zip(tgt_block_mask, tgt, ctx):
            # use only the target tokens that are included in the mask
            tgt_seq = tgt_seq[tgt_seq_mask]

            # use only the ctx tokens that are not padding
            ctx_seq = ctx_seq[ctx_seq.named_columns["image_ids"] != MASK_IMAGE_ID]

            # pads up to the prediction sequence length
            pad_amt = self.sequence_length_prediction - tgt_seq.sequence_length
            assert (
                pad_amt >= 0
            ), f"prediction sequence length {tgt_seq.sequence_length} too long by {-pad_amt}"
            if pad_amt > 0:
                tgt_seq = tgt_seq.pad(pad_amt, MASK_IMAGE_ID)

            tgt_seq_packed.append(tgt_seq)

        tgt_seq_packed = TensorSequence.stack(tgt_seq_packed)
        tgt_seq_packed = TensorSequence.cat([tgt_seq_packed, ctx])

        return tgt_seq_packed
