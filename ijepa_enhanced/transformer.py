from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        use_bias=False,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=use_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=use_bias)
        self.proj = nn.Linear(inner_dim, dim, bias=use_bias)

    def forward(self, x, attention_mask=None):
        """
        attention_mask: A boolean mask where a value of True indicates that the element should take part in attention
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, attention_mask)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, use_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            heads,
            dim_head,
            use_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, use_bias)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        gradient_checkpoint=False,
        use_bias=True,
    ):
        super().__init__()
        self.depth = depth
        self.layers = nn.Sequential(
            *[
                TransformerBlock(dim, heads, dim_head, mlp_dim, use_bias)
                for _ in range(depth)
            ]
        )
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
