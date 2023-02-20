import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.components.decoder import Decoder
from model.utils.mask import generate_mask
from model.metric import BLEU
from typing import Union, Callable
import pickle

import os

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class GPTModel(nn.Module):
    def __init__(self, vocab_size: int,  n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float,activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.decoder = Decoder(vocab_size=vocab_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.to(device)
    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        x = self.embedding_layer(x)
        output = self.decoder(x, mask, training)
        return output


class GPT:
    def __init__(self,
                vocab_size: int, 
                n: int = 12, 
                embedding_dim: int = 768, 
                heads: int = 12, 
                d_ff: int = 2048, 
                dropout_rate: float = 0.1, 
                eps: float = 1e-5, 
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                learning_rate: float = 0.0006,
                checkpoint: str = None):
        self.model = GPTModel(vocab_size=vocab_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.embedding_dim = embedding_dim
        self.checkpoint = checkpoint
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.metric = BLEU()
        self.criterion = nn.CrossEntropyLoss()

        self.entropy_loss = 0.0
        self.bleu_score = 0.0

        self.epoch = 0

    def build_dataset(self, inputs: torch.Tensor, labels: torch.Tensor = None, batch_size: int = 1, shuffle: bool = True):
        if labels is None:
            dataset = TensorDataset(inputs)
        else:
            dataset = TensorDataset(inputs, labels)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def __save_model(self, checkpoint: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'loss': self.entropy_loss
        }, checkpoint)

    def __load_model(self, checkpoint: str):
        checkpoint_data = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        self.epoch = checkpoint_data['epoch']
        self.entropy_loss = checkpoint_data['loss']
        self.model.eval()

    def save_model(self, path: str = None):
        if path is not None:
            self.__save_model(path)
            self.checkpoint = path
        elif self.checkpoint is not None:
            self.__save_model(self.checkpoint)
        else:
            print("Checkpoint not found")

    def load_model(self, path: str = None):
        if path is not None:
            self.__load_model(path)
            self.checkpoint = path
        elif self.checkpoint is not None:
            self.__load_model(self.checkpoint)
        else:
            print("Checkpoint not found")

        
    def cross_entropy_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)
        loss = 0.0
        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])

        loss = loss/batch_size

        return loss

    def pretrain_step(self, data: torch.Tensor):
        inputs = data[:, :-1]
        labels = data[:, 1:]

        _, look_ahead_mask = generate_mask(inputs)

        outputs = self.model(inputs, look_ahead_mask, True)

        _, preds = torch.max(outputs, dim=-1)

        self.bleu_score += self.metric.score(outputs=preds, labels=labels)

        self.entropy_loss += self.cross_entropy_loss(outputs=outputs, labels=labels)

    def fine_tune_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        _, look_ahead_mask = generate_mask(inputs)

        outputs = self.model(inputs, look_ahead_mask, True)

        _, preds = torch.max(outputs, dim=-1)

        self.bleu_score += self.metric.score(outputs=preds, labels=labels)

        self.entropy_loss += self.cross_entropy_loss(outputs=outputs, labels=labels)

    def fit(self, inputs: torch.Tensor, labels: torch.Tensor = None, batch_size: int = 1, epochs: int = 1, shuffle_data: bool = True, mini_batch: int = 1):
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
        
        dataloader = self.build_dataset(inputs=inputs, labels=labels, batch_size=batch_size, shuffle=shuffle_data)
        
        training_process = "Pretraininig"
        
        for _ in range(epochs):
            self.epoch += 1

            for index, data in enumerate(dataloader, 0):
                if labels is None:
                    self.pretrain_step(data=data[0].to(device))
                else:
                    self.fine_tune_step(inputs=data[0].to(device), labels=data[1].to(device))
                    training_process = "Fine-tunning"
                if index%(batch_size*mini_batch):
                    # Statiscal
                    print(f"{training_process} Epoch: {self.epoch} Batch: {index} Loss: {(self.entropy_loss/(batch_size*mini_batch)):.4f}")

                    # Set default
                    self.entropy_loss = 0.0
                    self.bleu_score = 0.0
        
        print("============Finished Training============")

    def __predict_token(self, input: torch.Tensor):
        _, look_ahead_mask = generate_mask(input)
        
        outputs = self.model(input, look_ahead_mask, False)

        predict = outputs[:, -1, :]
        _, token_id = torch.max(predict, dim=-1)

        return token_id


    def predict(self, data: torch.Tensor, limit_tokens: int, end_token: int):
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
        else:
            print("Model is not Trained")
            return None
        
        data = data.to(device)

        for _ in range(limit_tokens):
            token = self.__predict_token(input=data)

            if token == end_token:
                break
            
            data = torch.concat([data, token.unsqueeze(0)], dim=-1)

        return data
        




    