import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.components.decoder import Decoder
from model.utils.mask import generate_look_ahead_mask
from model.metric import BLEU
from model.loss import Perplexity
from typing import Union, Callable
from model.components.classifier import Classifier
from torchsummary import summary

import os

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class GPTModel(nn.Module):
    def __init__(self, 
                token_size: int,  
                n: int, 
                embedding_dim: int, 
                heads: int, 
                d_ff: int, 
                dropout_rate: float, 
                eps: float,
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=embedding_dim)
        self.decoder = Decoder(token_size=token_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        output = self.decoder(x, mask)
        return output



class GPT:
    def __init__(self,
                token_size: int,  
                n: int = 8, 
                embedding_dim: int = 512, 
                heads: int = 8, 
                d_ff: int = 2048, 
                dropout_rate: float = 0.1, 
                eps: float = 0.1,
                optimizer: optim.Optimizer = optim.Adam,
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                checkpoint: str = None) -> None:
        self.model = GPTModel(token_size=token_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.optimizer = optimizer(params=self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.loss = 0.0

        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def sumary(self):
        summary(self.model)

    def build_dataset(self, data: torch.Tensor, batch_size: int, shuffle: bool):
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
    
    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = labels.size(0)
        loss = 0.0

        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])
        loss = loss/batch_size

        return loss
    
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.optimizer.zero_grad()

        mask = generate_look_ahead_mask(inputs)

        outputs = self.model(inputs, mask)

        loss = self.calculate_loss(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.loss += loss.item()

    def save_model(self, path: str):
        decoder_layers_prams = []
        for layer in self.model.decoder.decoder_layers:
            decoder_layers_prams.append(layer.state_dict())
        torch.save({
            'module': self.model.state_dict(),
            'decoder_layers': decoder_layers_prams,
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['module'])
            for index, layer in enumerate(self.model.decoder.decoder_layers):
                layer.load_state_dict(checkpoint['decoder_layers'][index])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def fit(self, seq: torch.Tensor, learning_rate: float = 0.001, epochs: int = 1, batch_size: int = 1, shuffle: bool = True, mini_batch: int = 1, checkpoint: str = None):
        self.model.train()
        dataloader = self.build_dataset(seq, batch_size, shuffle)
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate
        
        total = len(dataloader)
        delta = total - (total//mini_batch)*mini_batch
        for epoch in range(epochs):
            for index, data in enumerate(dataloader):
                inputs = data[0][:, :-1].to(device)
                labels = data[0][:, 1:].to(device)
                self.train_step(inputs, labels)

                if index%mini_batch == mini_batch-1:
                    print(f"Epoch {epoch+1} Batch {index+1} Loss: {(self.loss/mini_batch):.4f}")
                    self.loss = 0.0
                elif index == total - 1:
                    print(f"Epoch {epoch+1} Batch {index+1} Loss: {(self.loss/delta):.4f}")
                    self.loss = 0.0

        
        if checkpoint is not None:
            self.checkpoint = checkpoint

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)

    def predict(self, seq: torch.Tensor, num_tokens: int, end_token: int):
        seq = seq.to(device)

        self.model.eval()

        for _ in range(num_tokens):
            look_ahead_mask = generate_look_ahead_mask(seq)
            output = self.model(seq, look_ahead_mask)

            predict = output[:, -1, :]
            _, token = torch.max(predict, dim=-1)

            if token == end_token:
                break

            seq = torch.cat([seq, token.unsqueeze(0)], dim=-1)

        return seq