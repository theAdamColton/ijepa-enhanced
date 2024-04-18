import einx
from torch import nn

from .transformer import Transformer
from .vit import PositionalEmbeddings


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=4,
        projection_dim=256,
        max_height=64,
        max_width=64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer = Transformer(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            hidden_size // num_attention_heads,
            intermediate_size,
        )
        self.pos_emb = PositionalEmbeddings(hidden_size, max_height, max_width)

    def forward(self, x, attention_mask, tgt_mask, position_ids):
        """
        x: All input patch data
        tgt_mask: Is True where the patch is a prediction target
        height_ids: height ids
        width_ids: width ids
        """
        # No information is permitted from the states to be predicted
        x = einx.multiply("b s z, b s -> b s z", x, ~tgt_mask)

        # TODO! should position information be given for tokens that are not being predicted?
        x = x + einx.multiply(
            "b s z, b s -> b s z",
            self.pos_emb(position_ids),
            tgt_mask,
        )

        x = self.transformer(x, attention_mask)

        return x
