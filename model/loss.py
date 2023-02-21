import torch

class Perplexity:
    def __init__(self) -> None:
        pass

    def loss(self, entropy_loss: torch.Tensor):
        return torch.exp(entropy_loss)