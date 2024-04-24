import torch
from torch import nn
import torch.nn.functional as F
import einx

from torch.utils.checkpoint import checkpoint


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, attention_mask=None):
        """
        attention_mask: A boolean mask where a value of True indicates that the element should take part in attention
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, attention_mask)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            heads,
            dim_head,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x, attention_mask=None):
        x = self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, gradient_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.layers = nn.Sequential(
            *[TransformerBlock(dim, heads, dim_head, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.gradient_checkpoint = gradient_checkpoint

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            if self.gradient_checkpoint:
                x = checkpoint(
                    layer,
                    x,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attention_mask)
        return x
