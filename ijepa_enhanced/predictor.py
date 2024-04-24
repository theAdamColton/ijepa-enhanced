import torch
import einx
from torch import nn

from .transformer import Transformer
from .vit import PositionalEmbeddings


class Predictor(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=768,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=4,
        projection_dim=256,
        projection_heads=4,
        max_height=64,
        max_width=64,
        gradient_checkpoint=False,
        name="",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_in = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
        )
        self.is_prediction_token = nn.Parameter(torch.randn(hidden_size))
        self.pos_emb = PositionalEmbeddings(hidden_size, max_height, max_width)
        self.transformer = Transformer(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            hidden_size // num_attention_heads,
            intermediate_size,
            gradient_checkpoint=gradient_checkpoint,
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.pred_head = nn.Linear(
            hidden_size, projection_dim * projection_heads, bias=False
        )
        self.projection_dim = projection_dim
        self.projection_heads = projection_heads

    def forward(self, x, attention_mask, tgt_mask, position_ids):
        """
        x: All input patch data
        tgt_mask: Is True where the patch is a prediction target
        position_ids: Contains:
            height_ids: height ids
            width_ids: width ids
        """

        x = self.proj_in(x)

        # No patch information is permitted from the states to be predicted
        # They are still allowed to be attended to
        x = einx.multiply("b s z, b s -> b s z", x, ~tgt_mask)

        # TODO! should position information be given for tokens that are not being predicted?
        # yes
        x = x + self.pos_emb(position_ids)

        # Add a special conditioning token for tokens that are the prediction target
        x = x + einx.multiply(
            "b s, z -> b s z",
            tgt_mask,
            self.is_prediction_token,
        )

        x = self.transformer(x, attention_mask)
        x = self.norm(x)
        x = self.pred_head(x)

        x = einx.rearrange("b s (h z) -> b s h z", x, h=self.projection_heads)

        return x
