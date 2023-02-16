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

        self.running_loss = 0.0
        self.bleu_score = 0.0
        self.running_accuracy = 0.0

        self.epoch = 0
    
    def fit(self, inputs: torch.Tensor, batch_size: int = 1, epochs: int = 1, show_info: int = 1):
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
        """ for param in self.model.parameters():
            param.requires_grad = False """
        
        # scheduler = ScheduledOptimizer(optimizer=optimizer, embedding_dim=self.embedding_dim, warmup_steps=4000)
        dataloader = self.build_dataset(inputs=inputs, batch_size=batch_size)
        for _ in range(epochs):

            self.epoch += 1

            for index, data in enumerate(dataloader, 0):
                # training process
                self.train_step(data)
                
                if index != 0 and index%show_info == 0:
                    print(f"Epoch: {self.epoch} Batch: {index} Loss: {(self.running_loss/show_info)}")
                    self.running_loss = 0.0
                    self.running_accuracy = 0.0
                    self.bleu_score = 0.0

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)

    def train_step(self, data: torch.Tensor):
        inputs = data[0][:, :-1].to(device)
        labels = data[0][:, 1:].to(device)
        # tasks = data[1]

        _, look_ahead_mask = generate_mask(inputs)

        self.optimizer.zero_grad()

        # Feed Forward Propagation
        with torch.set_grad_enabled(True):
            outputs = self.model(inputs, look_ahead_mask, True)
            loss = self.loss_function(outputs, labels)

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        # scheduler.step()

        # _, predicted = torch.max(outputs, dim=-1)

        self.running_loss += loss.item()
        # self.running_accuracy += self.accuracy_function(predicted, labels)
        # self.bleu_score += self.metric.score(outputs=predicted, labels=labels)

    def build_dataset(self, inputs: torch.Tensor, batch_size: int):
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader


    def accuracy_function(self, outputs: torch.Tensor, labels: torch.Tensor):
        same = torch.eq(labels, outputs).type(torch.int64)

        # mask = torch.logical_not(labels == 0).type(torch.int64)

        return same.sum()/(labels.size(0)*labels.size(1))

    def loss_function(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = labels.size(0)
        total_loss = 0.0

        mask = torch.logical_not(labels == 0).type(torch.int64)
        # tasks_loss = criterion(tasks_proba, tasks)
        
        for batch in range(batch_size):
            loss = self.criterion(outputs[batch], labels[batch])
            loss = loss*mask[batch]
            total_loss += loss.sum()/mask.sum()
        
        total_loss = (total_loss/(batch_size))
        # total_loss = total_loss*mask
        # print(mask.sum())

        return total_loss

    def save_model(self, path: str):
        with open(path, 'wb') as file:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.running_loss
            }, file)
        self.checkpoint = path
        print(f"Your Model Saved at {path}")

    def load_model(self, path: str):
        if os.path.exists(path) == True:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.running_loss = checkpoint['loss']


    def info(self):
        self.load_model(self.checkpoint)
        print("Model's State Dict: ")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        print("===================================")
        for param_tensor in self.model.decoder.state_dict():
            print(param_tensor, "\t", self.model.decoder.state_dict()[param_tensor].size())
        
    
    def predict(self, sequence: torch.Tensor, max_length: int, end_token: int):
        self.load_model(self.checkpoint)
        self.model.eval()
        sequence = sequence.to(device)

        for i in range(max_length):
            _, look_ahead_mask = generate_mask(sequence)
            look_ahead_mask = look_ahead_mask.to(device)
            
            with torch.no_grad():
                preds = self.model(sequence, look_ahead_mask, False)

            preds = preds[:, -1, :]

            _, predicted_id = torch.max(preds, dim=-1)
            if predicted_id == end_token:
                break
            if predicted_id == 0:
                continue
            sequence = torch.concat([sequence, predicted_id.unsqueeze(0)], dim=-1)

        return sequence


    def pretrain(self, sequences: torch.Tensor, batch_size: int, epochs: int = 1):
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
        
        dataset = TensorDataset(sequences)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            self.epoch += 1

            for index, data in enumerate(dataloader, 0):
                self.pretrain_step(data)

                if index%batch_size==0:
                    print(f"Epoch: {self.epoch} Loss: {self.running_loss}")
                    self.running_loss = 0.0

        
    def pretrain_step(self, data: torch.Tensor):
        inputs = data[0][:, :-1].to(device)
        targets = data[0][:, 1:].to(device)

        _, look_ahead_mask = generate_mask(inputs)

        with torch.set_grad_enabled(True):
            outputs = self.model(inputs, look_ahead_mask, True)
            loss = self.loss_function(outputs=outputs, labels=targets)

        loss.backward()
        self.optimizer.step()

        self.running_loss += loss.item()
        
            
    