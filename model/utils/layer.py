import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable

from model.utils.attention import MultiHeadAttention
from model.utils.ffn import PositionWiseFeedForward
from model.utils.res import ResidualConnection
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.ffn = PositionWiseFeedForward(d_ff=d_ff, embedding_dim=embedding_dim, activation=activation)

        self.residual_connection_1 = ResidualConnection(dropout_rate=dropout_rate)
        self.residual_connection_2 = ResidualConnection(dropout_rate=dropout_rate)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)

        self = self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # sublayer 1
        q = k = v = x
        attention_output = self.masked_multi_head_attention(q, k, v, mask)
        attention_output = self.residual_connection_1(attention_output, x)
        sublayer_1 = self.layer_norm_1(attention_output)

        # sublayer 2
        ffn_output = self.ffn(sublayer_1)
        ffn_output = self.residual_connection_2(ffn_output, sublayer_1)
        sublayer_2 = self.layer_norm_2(ffn_output)

        return sublayer_2

