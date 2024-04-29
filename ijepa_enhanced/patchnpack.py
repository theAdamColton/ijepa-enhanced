"""
This file implements a pure python/pytorch greedy patch n pack algorithm
"""

from typing import List
import torch
import einx

from tensorsequence import TensorSequence

from .utils import random_uniform


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
TENSORSET_PADDING_VALUE_DICT = {
    "image_ids": MASK_IMAGE_ID,
    "positions": 0,
    "patches": 0,
    "states": 0,
    "prediction_block_masks": 0,
    "prediction_mask": 0,
    "label": MASK_IMAGE_ID,
    "ids": MASK_IMAGE_ID,
}


def clamp_tensorsequence_to_length(
    sequence: List[TensorSequence],
    sequence_length: int,
    rng=None,
    pad_value_dict=TENSORSET_PADDING_VALUE_DICT,
) -> TensorSequence:
    """
    coerces a list of sequences into the specified sequence_length by padding and randomly dropping sequence items
    """
    sequence = TensorSequence.cat(sequence)

    # if the sequence length overflows, randomly drops items in the sequence
    needs_drop = sequence.sequence_length > sequence_length
    if needs_drop:
        mask = get_sample_mask(sequence.sequence_length, sequence_length, rng)
        sequence = sequence.iloc[mask]

    # if the sequence length is too short, pads
    pad_amt = sequence_length - sequence.sequence_length
    needs_pad = pad_amt > 0
    if needs_pad:
        sequence = sequence.pad(pad_amt, value_dict=pad_value_dict)

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


class Packer:
    def __init__(self, sequence_length, batch_size, rng=None):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.unpacked_sequences: List[TensorSequence] = []
        self.packed_sequences: List[TensorSequence] = []
        self.rng = rng

    def _flush(self):
        assert len(self.unpacked_sequences) > 1

        packed, self.unpacked_sequences = self.unpacked_sequences[:-1], [
            self.unpacked_sequences[-1]
        ]

        packed = clamp_tensorsequence_to_length(packed, self.sequence_length)
        self.packed_sequences.append(packed)

    def _needs_flush(self):
        if len(self.unpacked_sequences) < 2:
            return False
        s = sum(ts.sequence_length for ts in self.unpacked_sequences[:-1])
        if s > self.sequence_length:
            return True
        return False

    def _append(self, sequence: TensorSequence):
        self.unpacked_sequences.append(sequence)

    def append(self, sequence: TensorSequence):
        self._append(sequence)
        if self._needs_flush():
            self._flush()

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


class MakeIterableMixin:
    def make_iter(self, data_iter):
        """
        pass in a data iter which yields rows of data,
            must contain at least "pixel_values"

        returns a generator that yields batches
        """
        data_iter = iter(data_iter)
        while True:
            if not self.can_pop_batch():
                try:
                    row = next(data_iter)
                except StopIteration:
                    return
                image = row["pixel_values"]

                image_id = row.get("__id__")

                other_ids = []
                label = row.get("label")
                if label is not None:
                    other_ids.append(label)

                self.append_image(image, image_id=image_id, label=label)
                continue

            yield self.pop_batch()


class PatchNPacker(MakeIterableMixin):
    def __init__(self, patch_size, sequence_length, batch_size, rng=None):
        self.patch_size = patch_size
        self.packer = Packer(sequence_length, batch_size, rng)
        self.__id = 0

    def append_image(
        self,
        pixel_values,
        image_id=None,
        **named_metadata_ids,
    ):
        """
        Append an image to be packed, along with a unique identifier and any metadata ids

        Each additional named_metadata_id if specified has to be an integer
        """
        if image_id is None:
            image_id = self.__id
            self.__id += 1

        assert (
            image_id != MASK_IMAGE_ID
        ), f"{image_id} cannot be the same as the mask image id {MASK_IMAGE_ID}"

        patches, positions = patch(pixel_values, self.patch_size)

        s = len(patches)

        full = lambda x: torch.full((s,), x, dtype=torch.long, device=patches.device)
        image_ids = full(image_id)

        named_metadata_ids = {k: full(x) for k, x in named_metadata_ids.items()}

        named_columns = dict(
            patches=patches,
            positions=positions,
            image_ids=image_ids,
            **named_metadata_ids,
        )

        sequence = TensorSequence(named_columns=named_columns)

        self.packer.append(sequence)

    def can_pop_batch(self):
        return self.packer.can_pop_batch()

    def pop_batch(self):
        return self.packer.pop_batch()


class ContextTargetPatchNPacker(MakeIterableMixin):
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
        self.num_prediction_targets = num_prediction_targets

        self.context_packer = Packer(sequence_length_context, batch_size, rng)
        self.target_packer = Packer(sequence_length_target, batch_size, rng)

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.rng = rng
        self.__id = 0

    def append_image(
        self,
        pixel_values,
        image_id=None,
        **named_metadata_ids,
    ):
        _, h, w = pixel_values.shape
        nph = h // self.patch_size
        npw = w // self.patch_size

        if image_id is None:
            image_id = self.__id
            self.__id += 1

        patches, positions = patch(pixel_values, self.patch_size)
        sequence_length = patches.shape[0]
        if sequence_length > self.sequence_length_target:
            raise ValueError(
                f"image has too many patches {sequence_length} to be packed into size {self.sequence_length_target}"
            )

        # Pads the metadata ids to the full sequence length
        full = lambda x: torch.full(
            (sequence_length,), x, dtype=torch.long, device=pixel_values.device
        )
        image_ids = full(image_id)
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

        return self._append_sequence(sequence, nph, npw)

    def _append_sequence(
        self,
        sequence,
        nph,
        npw,
    ):
        # Samples 4 rectangular blocks that will be used for prediction loss
        prediction_block_masks = []
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

            prediction_block_masks.append(target_block)

        # shape: (S, self.num_prediction_targets)
        # the num_prediction_targets is the last dimension
        prediction_block_masks = torch.stack(prediction_block_masks, -1)

        # Samples 1 rectangular context block
        context_block = sample_rect_mask(
            nph, npw, 0.85, 1.0, 1.0, 1.0, rng=self.rng
        ).flatten()

        # target_any is a mask that is true if a patch is part of a target block
        # shape: (S,)
        target_any = prediction_block_masks.sum(-1) > 0

        # context is removed wherever the target mask is true,
        # this is to make it more difficult to predict the target from the context, because they don't overlap
        context_block = context_block & ~target_any

        # Only contains the patches in the context block
        context_sequence = sequence.iloc[context_block]

        # Adds a new columns to the sequence with the prediction block masks
        sequence.named_columns["prediction_block_masks"] = prediction_block_masks

        self.context_packer._append(context_sequence)
        self.target_packer._append(sequence)

        if self.context_packer._needs_flush() or self.target_packer._needs_flush():
            self.target_packer._flush()
            self.context_packer._flush()

    def can_pop_batch(self):
        return self.context_packer.can_pop_batch()

    def pop_batch(self):
        """
        returns a context batch and a target batch as a tuple
        each batch is a TensorSequence
        """
        if not self.can_pop_batch():
            return None

        return self.context_packer.pop_batch(), self.target_packer.pop_batch()

    def pack_prediction_target_sequence(
        self, target: TensorSequence, context: TensorSequence, prediction_block_mask
    ):
        device = target.all_columns[0].device
        b = target.leading_shape[0]

        packed = []
        for i in range(b):
            prediction_block_mask_sequence = prediction_block_mask[i]
            target_sequence = target.iloc[i]
            context_sequence = context.iloc[i]

            target_sequence = target_sequence.iloc[prediction_block_mask_sequence]

            context_sequence = context_sequence.iloc[
                context_sequence["image_ids"] != MASK_IMAGE_ID
            ]

            # TODO clamping to length might cause a problem if all of the prediction targets are randomly dropped
            # which is possible
            packed_sequence = clamp_tensorsequence_to_length(
                [target_sequence, context_sequence],
                self.sequence_length_prediction,
            )
            packed.append(packed_sequence)

        packed = TensorSequence.stack(packed)

        return packed
