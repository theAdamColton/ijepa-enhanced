"""
Lookup free quantization
from https://arxiv.org/abs/2310.05737
"""

from math import log2, ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch import nn
import einx


def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    return x * y[..., *[None for _ in range(ndim_to_expand)]]


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benifit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.
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
        avg_probs = masked_mean(probs, mask)
        avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = einx.mean("... D -> D", probs)

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        sample_entropy = masked_mean(sample_entropy, mask).mean()
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

    @property
    def dtype(self):
        return self.project_in.weight.dtype

    def encode(self, x):
        x = self.project_in(x)

        # split out number of codebooks

        x = einx.rearrange("b n (c d) -> b n c d", x, c=self.num_codebooks)

        x = self._quantize(x)

        indices = einx.sum((x > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")

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
        self,
        x,
        mask=None,
    ):
        # entropy aux loss

        if self.training:
            # the same as euclidean distance up to a constant
            distance = -2 * einsum(
                "... i d, j d -> ... i j", original_input, self.codebook
            )

            prob = (-distance * inv_temperature).softmax(dim=-1)

            # account for mask

            if exists(mask):
                prob = prob[mask]
            else:
                prob = rearrange(prob, "b n ... -> (b n) ...")

            # whether to only use a fraction of probs, for reducing memory

            if self.frac_per_sample_entropy < 1.0:
                num_tokens = prob.shape[0]
                num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                rand_mask = torch.randn(num_tokens).argsort(dim=-1) < num_sampled_tokens
                per_sample_probs = prob[rand_mask]
            else:
                per_sample_probs = prob

            # calculate per sample entropy

            per_sample_entropy = entropy(per_sample_probs).mean()

            # distribution over all available tokens in the batch

            avg_prob = reduce(per_sample_probs, "... c d -> c d", "mean")
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

            entropy_aux_loss = (
                per_sample_entropy - self.diversity_gamma * codebook_entropy
            )
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(
                original_input, quantized.detach(), reduction="none"
            )

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero

        # merge back codebook dim

        x = rearrange(x, "b n c d -> b n (c d)")

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # complete aux loss

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight
            + commit_loss * self.commitment_loss_weight
        )

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)
