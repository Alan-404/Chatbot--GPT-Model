import torch
import torch.nn as nn

""" from typing import Union, Callable """
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x