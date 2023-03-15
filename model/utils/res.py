import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate: float) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self = self.to(device)
    
    def forward(self, tensor: torch.Tensor, pre_tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.dropout_layer(tensor)

        tensor = tensor + pre_tensor

        return tensor

    