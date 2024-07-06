from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from ijepa_enhanced.data import MASK_ID
from ijepa_enhanced.tome.tome import TokenMerger


def _init_weights(m):
    std = 0.02
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MLP(nn.Module):
    def __init__(
        self,
        num_features,
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(num_features * 4, num_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def make_attn_mask(sequence_ids, sequence_mask, num_attention_heads):
    attn_mask = sequence_ids.unsqueeze(2) != sequence_ids.unsqueeze(1)
    if sequence_mask is not None:
        attn_mask = attn_mask | sequence_mask.unsqueeze(2) | sequence_mask.unsqueeze(1)
    attn_mask = attn_mask.unsqueeze(1).repeat(
        1,
        num_attention_heads,
        1,
        1,
    )  # b s s -> b h s s
    return attn_mask


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads=8,
        qkv_bias=True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None, return_keys=False):
        """
        attn_mask: Contains True where a i,j pair is NOT allowed to be attended to
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_attention_heads, C // self.num_attention_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        if attn_mask is not None:
            neg_inf = torch.finfo(attn.dtype).min
            attn.masked_fill_(attn_mask, neg_inf)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if return_keys:
            return x, k

        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(1e-5 * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        qkv_bias=True,
        use_layerscale=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ls1 = LayerScale(dim) if use_layerscale else nn.Identity()
        self.attn = Attention(
            dim,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        self.ls2 = LayerScale(dim) if use_layerscale else nn.Identity()

    def forward(self, x, attn_mask=None):
        """
        attn_mask: Contains True where a i,j pair is NOT allowed to be attended to
        """
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))

        return x


class JointPositionEmbedding(nn.Module):
    def __init__(self, dim=512, max_height=32, max_width=32):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.embed = nn.Parameter(torch.zeros(max_height * max_width, dim))
        trunc_normal_(self.embed, std=0.02)

    def forward(self, height_ids, width_ids):
        ids = height_ids * self.max_width + width_ids
        return self.embed[ids]


class IndependentPositionEmbedding(nn.Module):
    def __init__(self, dim=512, max_height=32, max_width=32):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.h_embed = nn.Parameter(torch.zeros(max_height, dim))
        self.w_embed = nn.Parameter(torch.zeros(max_width, dim))
        trunc_normal_(self.h_embed, std=0.02)
        trunc_normal_(self.w_embed, std=0.02)

    def forward(self, height_ids, width_ids):
        return self.h_embed[height_ids] + self.w_embed[width_ids]


class TomeBlocks(nn.Module):
    def __init__(
        self,
        depth=12,
        dim=512,
        num_attention_heads=8,
        qkv_bias=True,
        use_layerscale=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(dim, num_attention_heads, qkv_bias, use_layerscale)
                for _ in range(depth)
            ]
        )
        self.num_attention_heads = num_attention_heads

    def forward(self, x, sequence_ids, sequence_mask, r: int = 0, mode: str = "drop"):
        """
        sequence_ids: long tensor of shape (batch, sequence_length)
        sequence_mask: Optional bool tensor of shape (batch, sequence_length)
         contains True where a token is NOT allowed to be attended to.
        r: int, The amount of tokens to merge per layer
        mode: str, The tome merging mode

        Returns:
            hidden_states: Tensor (batch, sequence_length - num_layers * r, -1)
            tome: Optional[TokenMerger]
        """
        attn_mask = make_attn_mask(
            sequence_ids, sequence_mask, self.num_attention_heads
        )
        if sequence_mask is not None:
            # fills with MASK ID
            # in practice, this allows TOME to merge more of the mask ids before
            # merging non padding
            sequence_ids.masked_fill_(sequence_mask, torch.tensor(MASK_ID))

        tome = None
        tome_enabled = r > 0

        for blk in self.blocks:
            if tome_enabled:
                z, k = blk.attn(blk.norm1(x), attn_mask=attn_mask, return_keys=True)
                z = blk.ls1(z)
                x = x + z

                k = k.mean(1)
                if tome is None:
                    tome = TokenMerger(k, r, sequence_ids=sequence_ids, mask_id=MASK_ID)
                else:
                    tome = TokenMerger(
                        k,
                        r,
                        adm=tome.adm,
                        sequence_ids=sequence_ids,
                        mask_id=MASK_ID,
                    )

                sequence_ids = tome.merged_ids
                if sequence_mask is not None:
                    sequence_mask = tome.merge(sequence_mask, "mean") > 0
                attn_mask = make_attn_mask(
                    sequence_ids, sequence_mask, self.num_attention_heads
                )

                x = tome.merge(x, mode)

                x = x + blk.ls2(blk.mlp(blk.norm2(x)))
            else:
                x = blk(x, attn_mask)

        return x, tome


@dataclass
class PredictorOutput:
    hidden_states: torch.Tensor
    tome: Optional[TokenMerger]


class Predictor(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=384,
        depth=6,
        num_attention_heads=12,
        max_height=48,
        max_width=48,
        use_layerscale=True,
        use_joint_position_embed=False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.predictor_embed = nn.Linear(input_size, hidden_size, bias=True)
        if use_joint_position_embed:
            self.position_embed = JointPositionEmbedding(
                hidden_size, max_height, max_width
            )
        else:
            self.position_embed = IndependentPositionEmbedding(
                hidden_size, max_height, max_width
            )
        self.mask_token = nn.Parameter(torch.zeros(hidden_size))

        self.blocks = TomeBlocks(
            depth=depth,
            dim=hidden_size,
            num_attention_heads=num_attention_heads,
            use_layerscale=use_layerscale,
        )
        self.predictor_norm = nn.LayerNorm(hidden_size)
        self.predictor_proj = nn.Linear(hidden_size, input_size, bias=True)

        self.apply(_init_weights)
        trunc_normal_(self.mask_token, std=0.02)

        def rescale(param, layer_id):
            param.div_((2.0 * layer_id) ** 0.5)

        for i, layer in enumerate(self.blocks.blocks):
            rescale(layer.attn.proj.weight.data, i + 1)
            rescale(layer.mlp.fc2.weight.data, i + 1)

    def forward(
        self,
        x,
        height_ids,
        width_ids,
        target_mask,
        sequence_ids,
        sequence_mask=None,
        r: int = 0,
    ):
        """
        x: float tensor of shape (batch, sequence length, input_dim)
        height_ids: long tensor of shape (batch, sequence_length)
        width_ids: long tensor of shape (batch, sequence_length)
        target_mask: bool tensor of shape (batch, sequence_length)
         contains True where a token is a prediction target
        sequence_ids: long tensor of shape (batch, sequence_length)
        sequence_mask: Optional bool tensor of shape (batch, sequence_length)
         contains True where a token is NOT allowed to be attended to.
        r: int, The amount of tokens to merge per layer
        """

        x = self.predictor_embed(x)

        target_mask = target_mask.unsqueeze(-1)
        x = (x * ~target_mask) + self.mask_token.unsqueeze(0).unsqueeze(0) * target_mask
        x = x + self.position_embed(height_ids, width_ids)

        x, tome = self.blocks(x, sequence_ids, sequence_mask, r)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return PredictorOutput(x, tome)


class PredictorPositionless(nn.Module):
    """
    Same as Predictor but doesnt have it's own position embedding table
    """

    def __init__(
        self,
        input_size=768,
        hidden_size=384,
        depth=6,
        num_attention_heads=12,
        max_height=48,
        max_width=48,
        use_layerscale=True,
        use_joint_position_embed=False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.predictor_embed = nn.Linear(input_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(hidden_size))

        self.blocks = TomeBlocks(
            depth=depth,
            dim=hidden_size,
            num_attention_heads=num_attention_heads,
            use_layerscale=use_layerscale,
        )
        self.predictor_norm = nn.LayerNorm(hidden_size)
        self.predictor_proj = nn.Linear(hidden_size, input_size, bias=True)

        self.apply(_init_weights)
        trunc_normal_(self.mask_token, std=0.02)

        def rescale(param, layer_id):
            param.div_((2.0 * layer_id) ** 0.5)

        for i, layer in enumerate(self.blocks.blocks):
            rescale(layer.attn.proj.weight.data, i + 1)
            rescale(layer.mlp.fc2.weight.data, i + 1)

    def forward(
        self,
        x,
        position_embeddings,
        target_mask,
        sequence_ids,
        sequence_mask=None,
        r: int = 0,
    ):
        """
        x: float tensor of shape (batch, sequence length, input_dim)
        position_embeddings: float tensor of shape (batch, sequence length, input_dim)
        target_mask: bool tensor of shape (batch, sequence_length)
         contains True where a token is a prediction target
        sequence_ids: long tensor of shape (batch, sequence_length)
        sequence_mask: Optional bool tensor of shape (batch, sequence_length)
         contains True where a token is NOT allowed to be attended to.
        r: int, The amount of tokens to merge per layer
        """
        x = x + position_embeddings

        x = self.predictor_embed(x)

        target_mask = target_mask.unsqueeze(-1)
        x = (x * ~target_mask) + self.mask_token.unsqueeze(0).unsqueeze(0) * target_mask

        x, tome = self.blocks(x, sequence_ids, sequence_mask, r)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return PredictorOutput(x, tome)


@dataclass
class ViTOutput:
    hidden_states: torch.Tensor
    position_embeds: torch.Tensor
    tome: Optional[TokenMerger]


class ViT(nn.Module):
    def __init__(
        self,
        patch_size=16,
        image_channels=3,
        hidden_size=768,
        depth=12,
        num_attention_heads=12,
        max_height=48,
        max_width=48,
        use_layerscale=True,
        use_joint_position_embed=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.patch_embed = nn.Linear(patch_size**2 * image_channels, hidden_size)
        self.max_height = max_height
        self.max_width = max_width
        self.use_layerscale = use_layerscale
        self.use_joint_position_embed = use_joint_position_embed

        if use_joint_position_embed:
            self.position_embed = JointPositionEmbedding(
                hidden_size, max_height, max_width
            )
        else:
            self.position_embed = IndependentPositionEmbedding(
                hidden_size, max_height, max_width
            )
        self.blocks = TomeBlocks(
            depth=depth,
            dim=hidden_size,
            num_attention_heads=num_attention_heads,
            use_layerscale=use_layerscale,
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.apply(_init_weights)

        def rescale(param, layer_id):
            param.div_((2.0 * layer_id) ** 0.5)

        for i, layer in enumerate(self.blocks.blocks):
            rescale(layer.attn.proj.weight.data, i + 1)
            rescale(layer.mlp.fc2.weight.data, i + 1)

    def forward(
        self,
        x,
        height_ids,
        width_ids,
        sequence_ids,
        sequence_mask=None,
        r: int = 0,
        mode: str = "drop",
    ):
        """
        x: float tensor of shape (b s z) where z equals patch_size ** 2 * image_channels
        height_ids: long tensor of shape (batch, sequence_length)
        width_ids: long tensor of shape (batch, sequence_length)
        target_ids: long tensor of shape (batch, num_targets)
        sequence_ids: long tensor of shape (batch, sequence_length)
        sequence_mask: Optional bool tensor of shape (batch, sequence_length)
         contains True where a token is NOT allowed to be attended to.
        r: int, The amount of tokens to merge per layer
        mode: str, The tome merging mode
        """
        x = self.patch_embed(x)
        pos = self.position_embed(height_ids, width_ids)
        x = x + pos
        x, tome = self.blocks(
            x,
            sequence_ids,
            sequence_mask,
            r,
            mode,
        )
        x = self.norm(x)

        return ViTOutput(x, pos, tome)
