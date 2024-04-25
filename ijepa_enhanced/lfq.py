"""
Lookup free quantization
from https://arxiv.org/abs/2310.05737

adapted from parts of https://github.com/lucidrains/vector-quantize-pytorch/blob/ce3433256e40328de4a5bf06fadfcda2a227e696/vector_quantize_pytorch/lookup_free_quantization.py
"""

from math import log2, ceil
from typing import OrderedDict

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch import nn
import einx


def calculate_perplexity(codes, codebook_size, null_index=-1):
    """
    codes: Longtensor

    Perplexity is 2^(H(p)) where H(p) is the entropy over the codebook likelyhood

    the null index is assumed to be -1, perplexity is only calculated over the
    non null codes
    """
    dtype, device = codes.dtype, codes.device
    codes = codes.flatten()
    codes = codes[codes != null_index]
    src = torch.ones_like(codes)
    counts = torch.zeros(codebook_size).to(dtype).to(device)
    counts = counts.scatter_add_(0, codes, src)

    probs = counts / codes.numel()
    # Entropy H(x) when p(x)=0 is defined as 0
    logits = torch.log2(probs)
    logits[probs == 0.0] = 0.0
    entropy = -torch.sum(probs * logits)
    return 2**entropy


def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))


def entropy_loss(
    logits,
    mask=None,
    temperature=0.1,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-7,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        avg_probs = einx.mean("... D -> D", probs[mask])
        # avg_probs = masked_mean(probs, mask)
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = einx.mean("... D -> D", probs)

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        sample_entropy = sample_entropy[mask].mean()
        # sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return loss


class LFQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        temperature=0.1,
        eps=1e-9,
        name="",
    ):
        """
        dim: Integer dimension of input features
        codebook_size: Integer number of total number of codebook codes. Each vector of dimension `dim` will be
            quantized using `num_codebooks` codes, each code being an integer from 0 to `codebook_size`
        """

        super().__init__()
        assert log2(codebook_size).is_integer()
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks

        self.project_in = nn.Linear(dim, codebook_dims)
        self.project_out = nn.Linear(codebook_dims, dim)

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        self.temperature = temperature
        self.eps = eps

        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer("codebook", codebook, persistent=False)

    @property
    def dtype(self):
        return self.project_in.weight.dtype

    def _quantize(
        self,
        x,
    ):
        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value)
        if self.training:
            x = x + (quantized - x).detach()
        else:
            x = quantized
        return x

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.codebook_dim
        indices = 2 ** torch.arange(
            0,
            self.codebook_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # to some sort of float
        x = x.to(self.dtype)
        # -1 or 1
        x = x * 2 - 1
        x = einx.rearrange("... NC Z-> ... (NC Z)", x)
        x = self.project_out(x)
        return x

    def forward(
        self, x, mask=None, return_dict=None, return_indices=None, return_losses=None
    ):
        """
        mask: optional, is True where data is, is False where padding is
        """
        x = self.project_in(x)
        x = einx.rearrange("... (c d) -> ... c d", x, c=self.num_codebooks)
        original_x = x
        x = self._quantize(x)

        ret_dict = OrderedDict({"hidden_states": None})

        if return_indices:
            indices = self.bits_to_indices(x > 0)
            ret_dict["indices"] = indices

        if return_losses:
            # logits = 2 * torch.einsum(
            #     "... i d, j d -> ... i j",
            #     original_x,
            #     self.codebook,
            # )
            logits = torch.stack((original_x, -original_x), -1)
            # logits = original_x.unsqueeze(-1)
            loss = entropy_loss(
                logits,
                mask,
                self.temperature,
                self.sample_minimization_weight,
                self.batch_maximization_weight,
                self.eps,
            )
            ret_dict["entropy_loss"] = loss

            # commit loss applied between the tensors before and after the straight-through step
            commit_loss = F.mse_loss(original_x, x.detach(), reduction="none")
            if mask is not None:
                # commit_loss = masked_mean(commit_loss, mask)
                commit_loss = commit_loss[mask]
            commit_loss = commit_loss.mean()
            ret_dict["commit_loss"] = commit_loss

        x = einx.rearrange("... c d -> ... (c d)", x)
        x = self.project_out(x)

        ret_dict["hidden_states"] = x

        if return_dict:
            return ret_dict

        return tuple(ret_dict.values())
