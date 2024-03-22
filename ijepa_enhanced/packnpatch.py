from typing import List, Optional, Sequence, Sized
from dataclasses import dataclass
from torch import nn
import torch
import einx


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
    x: An image in (... C H W) form

    returns:
        patches: (... S Z)
        positions (... S 2)
          h,w positions of the patches
    """
    *_, h, w = x.shape
    assert h % patch_size == 0, h
    assert w % patch_size == 0, w
    x = einx.rearrange(
        "... C (NPH PH) (NPW PW) -> ... (NPH NPW) (C PH PW)",
        x,
        PH=patch_size,
        PW=patch_size,
    )
    nph = h // patch_size
    npw = w // patch_size
    device = x.device
    positions = torch.meshgrid(
        torch.arange(nph, device=device),
        torch.arange(npw, device=device),
        indexing="ij",
    )
    positions = torch.stack(positions, -1)
    positions = einx.rearrange("... NPH NPW Z -> ... (NPH NPW) Z", positions)
    return x, positions


class Packer:
    """
    Packs columns into batches based on sequence length
    """

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        # list of columns, each column contains rows of tensors to be batched
        self.batched_columns = []
        self.unbatched_columns = []

    @property
    def cached_batch_size(self):
        return len(self.batched_columns[0]) if self.batched_columns else 0

    def pack(self, *columns):
        """
        Greedily pack multiple columns into batches

        If row has a sequence length larger than sequence length, it is randomly downsampled.

        columns: A list of columns
            Each column contains a list of tensors
            Each tensor is a sequence, shape (S ...)
            where the trailing dimensions are the same and the
            dimension S can be different across different items

            Each row has the same sequence length

        """

        if not self.batched_columns:
            self.batched_columns = [[] for _ in range(len(columns))]
            self.unbatched_columns = [[] for _ in range(len(columns))]

        # Downsamples rows that have a sequence length that is too long
        n, m = len(columns[0]), len(columns)
        for i in range(n):
            mask = None
            for j in range(m):
                col = columns[j][i]
                s = len(col)
                if mask is not None:
                    columns[j][i] = columns[j][i][mask]
                elif s > self.sequence_length:
                    mask = get_sample_mask(col, self.sequence_length)
                    columns[j][i] = columns[j][i][mask]

        for j in range(len(columns)):
            to_pack = self.unbatched_columns[j] + columns[j]
            batched, leftover = pack(to_pack, self.sequence_length)
            self.batched_columns[j].extend(batched)
            self.unbatched_columns[j] = leftover

    def flush_batched_tensors(self, batch_size: Optional[int] = None):
        """
        returns either:
            a list of tensors each with batch size batch_size
            or None if there aren't enough items to make the batch size
        """
        if batch_size is None:
            batch_size = self.cached_batch_size

        if self.cached_batch_size < batch_size or self.cached_batch_size == 0:
            return None

        columns = []
        for j in range(len(self.batched_columns)):
            column, self.batched_columns[j] = (
                self.batched_columns[j][:batch_size],
                self.batched_columns[j][batch_size:],
            )
            column = torch.stack(column)
            columns.append(column)
        return columns


def cat_and_pad(tensors: List[torch.Tensor], sequence_length, pad_value=0):
    """
    concatenates and pads tensors to form a sequence of length exactly sequence_length
    """
    s = sum(len(x) for x in tensors)

    if s < sequence_length:
        pad_amt = max(sequence_length - s, 0)
        rest_shape = tensors[0].shape[1:]
        pad = torch.full(
            (pad_amt, *rest_shape),
            pad_value,
            device=tensors[0].device,
            dtype=tensors[0].dtype,
        )
        tensors.append(pad)
    tensors = torch.cat(tensors, 0)
    return tensors


def pack(sequences: List[torch.Tensor], sequence_length: int, pad_value=-1):
    """
    Greedily packs sequences into sequences of specified length

    sequences: a list of variable lengthed sequences, each sequence is:
        (S ...) where S is different across different items,
        and the trailing dimensions are the same across different items

    Returns:
        batches: A list of batches which may be empty

        unbatched_sequences: A list of leftover sequences that were
        not batched because they don't fill an entire batch
    """
    batches = []
    unbatched_sequences = []
    for sequence in sequences:
        unbatched_sequences.append(sequence)

        if sum(len(x) for x in unbatched_sequences) >= sequence_length:
            sequence = cat_and_pad(unbatched_sequences, sequence_length, pad_value)
            batches.append(sequence)
            unbatched_sequences = []

    return batches, unbatched_sequences


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


def get_sample_mask(x: torch.Tensor, max_length: int):
    s, *_ = x.shape
    u = torch.rand(s, device=x.device)
    p = max_length / s
    q = u.quantile(p)
    mask = u < q
    return mask


def sample_patches_target(x: torch.Tensor, num_targets: int):
    """
    x: A 2d array of image patches
        (... NPH NPW Z)
    """


def sample_patches_context(x: torch.Tensor, mask: torch.BoolTensor):
    """
    x: A 2d array of image patches
        (... NPH NPW Z)
    mask: a bool tensor where context is not allowed to sample patches
        (... NPH NPW)
    """


@dataclass
class PatchConfig:
    batch_size: int
    patch_size: int
    sequence_length: int
    num_targets: int = 4


class PackNPatchPipe:
    def __init__(self, image_iter, config: PatchConfig):
        self.image_iter = image_iter
        self.cropper = CropToMultipleOf(config.patch_size)
        self.config = config
        self.__id = 0

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self.image_iter)
        x = row["pixel_values"]
        id = row.get("id") or self.__id
        self.__id += 1
        x, positions = patch(x, self.config.patch_size)


class PackNPatchPipeTraining(PackNPatchPipe):
    def __next__(self):
        x = next(self.image_iter)
        x = patch(x, self.config.patch_size)
