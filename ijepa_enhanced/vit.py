"""
Some parts adapted from simple-vit from https://github.com/lucidrains/vit-pytorch/blob/5578ac472faf3903d4739ba783f3875b77177e57/vit_pytorch/simple_vit.py
"""

import torch
from torch import nn

from .transformer import Transformer


class PositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_height, max_width):
        """
        dim: hidden dimension
        max_h: max patch height position
        max_w: max patch width position
        """
        super().__init__()
        self.h_emb = nn.Embedding(max_height, dim)
        self.w_emb = nn.Embedding(max_width, dim)

    def forward(self, position_ids):
        h_indices, w_indices = position_ids.unbind(-1)
        return self.h_emb(h_indices) + self.w_emb(w_indices)


class ViT(nn.Module):
    """
    This differs from a usual vit in two ways:
    1.) Input processing
    There is no conv2d layer to process input patches.
    The input patches are simplyt projected using a linear projection.
    There is a norm before, and after this linear projection.
    Input pixels have their statistics tracked per patch position.

    2.) Position processing
    This ViT uses factorized absolute positional embeddings.
    There are seperate embedding tables for height and width positions.
    The two embeddings are combined using addition.

    The transformer block is exactly the same as a usual vit.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        patch_size: int = 32,
        image_channels: int = 3,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dim_head: int = 64,
        max_height: int = 64,
        max_width: int = 64,
        gradient_checkpoint=False,
        name="",
        use_bias=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        patch_dim = patch_size**2 * image_channels
        hidden_size = hidden_size

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size, bias=use_bias),
        )

        self.pos_emb = PositionalEmbeddings(hidden_size, max_height, max_width)

        self.transformer = Transformer(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            dim_head,
            intermediate_size,
            gradient_checkpoint=gradient_checkpoint,
            use_bias=use_bias,
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_indices: torch.LongTensor,
    ):
        """
        x: Image patches, shape (... patch_dim)
        height_indices: Height indices of the patch positions
        width_indices: Width indices of the patch positions

        Returns unnormalized tensor x
        """

        x = self.to_patch_embedding(x)
        x = x + self.pos_emb(position_indices)
        x = self.transformer(x, attention_mask)
        x = self.norm(x)
        return x
