import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.components.decoder import Decoder
from model.utils.mask import generate_mask
from model.metric import BLEU
from typing import Union, Callable
from model.optimizer import ScheduledOptimizer

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
                eps: float = 0.1,
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                checkpoint: str = None):
        self.model = GPTModel(vocab_size=vocab_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.embedding_dim = embedding_dim
        self.checkpoint = checkpoint
        self.metric = BLEU()
    
    def fit(self, train_data: torch.Tensor, batch_size: int = 1, epochs: int = 1, show_info: int = 1):
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

        optimizer = optim.Adam(params=self.model.parameters())
        scheduler = ScheduledOptimizer(optimizer=optimizer, embedding_dim=self.embedding_dim, warmup_steps=4000)
        dataloader = self.build_dataset(inputs=train_data, batch_size=batch_size)

        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            bleu_score = 0.0
            # accuracy_task = 0.0

            for index, data in enumerate(dataloader, 0):
                labels = data[0][:, 1:]
                inputs = data[0][:, :-1]

                # tasks = data[1]

                _, look_ahead_mask = generate_mask(inputs)

                inputs = inputs.to(device)
                labels = labels.to(device)
                # tasks = tasks.type(torch.LongTensor).to(device)
                look_ahead_mask = look_ahead_mask.to(device)

                # optimizer.zero_grad()
                scheduler.zero_grad()

                outputs = self.model(inputs, look_ahead_mask, True)

                # task_proba = outputs[1][:, -1, :]

                loss = self.loss_function(outputs, labels)

                loss.backward()
                # optimizer.step()
                scheduler.step()

                _, predicted = torch.max(outputs, dim=-1)

                running_loss += loss.item()
                running_accuracy += self.accuracy_function(predicted, labels)
                bleu_score += self.metric.score(outputs=predicted, labels=labels)
                # accuracy_task += self.accuracy_function_task(outputs=task_proba, labels=task
                
                if index != 0 and index%show_info == 0:
                    print(f"Epoch: {epoch + 1} Batch: {index} Loss: {(running_loss/show_info):.2f} Accuracy: {(running_accuracy/show_info):.2f}% BLEU: {(bleu_score/(show_info)):.2f}%")
                    running_loss = 0.0
                    running_accuracy = 0.0
                    bleu_score = 0.0
                    # accuracy_task = 0.0

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)

    
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

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        # mask = torch.logical_not(labels == 0).type(torch.int64)
        # tasks_loss = criterion(tasks_proba, tasks)
        
        for batch in range(batch_size):
            loss = criterion(outputs[batch], labels[batch])
            total_loss += loss
        
        total_loss = (total_loss/(batch_size))
        # total_loss = total_loss*mask
        # print(mask.sum())

        return total_loss

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, path)
        print(f"Your Model Saved at {path}")

    def load_model(self, path: str):
        if os.path.exists(path) == True:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def info(self):
        self.load_model(self.checkpoint)
        print("Model's State Dict: ")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        print("===================================")
    
    
    def predict(self, sequence: torch.Tensor, max_length: int, end_token: int):
        self.load_model(self.checkpoint)
        sequence = sequence.to(device)

        length = sequence.size(1)

        for i in range(max_length):
            _, look_ahead_mask = generate_mask(sequence)
            look_ahead_mask = look_ahead_mask.to(device)

            preds = self.model(sequence, look_ahead_mask, False)

            preds = preds[:, -1, :]

            predicted_id = torch.max(preds, dim=-1)

            if predicted_id == end_token:
                break

            sequence = torch.concat([sequence, predicted_id], dim=-1)

        return sequence[:, length: ]
    