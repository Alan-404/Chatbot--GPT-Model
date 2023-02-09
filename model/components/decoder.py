import torch
import torch.nn as nn
import torch.nn.functional as F
from typing  import Union, Callable
from model.utils.layer import DecoderLayer
from model.utils.postion import PositionalEncoding
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding()
        self.decoder_layers = [DecoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        # self.linear_task = nn.Linear(in_features=embedding_dim, out_features=task_size)
        self.embedding_dim = embedding_dim
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        x = x + self.positional_encoding.generate_position(embedding_dim=self.embedding_dim, length=x.size(1)).to(device)
        for layer in self.decoder_layers:
            x = layer(x, mask, training)

        text = self.linear(x)
        
        # task = self.linear_task(x)

        text = F.softmax(text, dim=-1)
        # task = F.softmax(task, dim=-1)

        return text