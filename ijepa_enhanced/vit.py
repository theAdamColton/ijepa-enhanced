"""
Some parts adapted from simple-vit from https://github.com/lucidrains/vit-pytorch/blob/5578ac472faf3903d4739ba783f3875b77177e57/vit_pytorch/simple_vit.py
"""

import torch
from torch import nn
import torch.nn.functional as F
import einx
from dataclasses import dataclass


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: einx.rearrange("b n (h d) -> b h n d", t, h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = einx.rearrange("b h n d -> b n (h d)", out)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


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

    def forward(self, h_indices, w_indices):
        return self.h_emb(h_indices) + self.w_emb(w_indices)


class ViT(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        patch_size: int = 32,
        image_channels: int = 3,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dim_head: int = 64,
        max_height: int = 32,
        max_width: int = 32,
    ):
        super().__init__()

        patch_dim = patch_size**2 * image_channels
        hidden_size = hidden_size

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.pos_emb = PositionalEmbeddings(hidden_size, max_height, max_width)

        self.model = Transformer(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            dim_head,
            intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        height_indices: torch.LongTensor,
        width_indices: torch.LongTensor,
    ):
        """
        x: Image patches, shape (... patch_dim)
        height_indices: Height indices of the patch positions
        width_indices: Width indices of the patch positions
        """

        x = self.to_patch_embedding(x)
        x = x + self.pos_emb(height_indices, width_indices)
        x = self.transformer(x)
        return x
