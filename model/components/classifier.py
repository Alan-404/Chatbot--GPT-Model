import torch
import torch.nn as nn

""" from typing import Union, Callable """
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class Classifier(nn.Module):
    def __init__(self, token_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=token_size, out_features=token_size)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)

        return x