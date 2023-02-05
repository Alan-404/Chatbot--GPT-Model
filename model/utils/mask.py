import torch
import numpy as np

def generate_padding_mask(tensor: torch.Tensor):
    return torch.Tensor(tensor == 0).type(torch.int64)[:, np.newaxis, np.newaxis, :]

def generate_look_ahead_mask(length: int):
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_mask(tensor: torch.Tensor):
    padding_mask = generate_padding_mask(tensor)

    look_ahead_mask = generate_look_ahead_mask(tensor.size(1))

    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask)

    return padding_mask, look_ahead_mask