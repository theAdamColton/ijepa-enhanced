from torch import nn

from .transformer import Transformer


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=4,
        projection_dim=256,
    ):
        super().__init__()
        self.transformer = Transformer(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            hidden_size // num_attention_heads,
            intermediate_size,
        )
        self.cls_head = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(self, x):
        x = self.transformer(x)
        x = self.cls_head(x)
        return x
