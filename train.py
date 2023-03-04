import torch
from argparse import ArgumentParser
from typing import Union, Callable
import torch.optim as optim
from model.gpt import GPT
from preprocessing.text import TextProcessor
parser = ArgumentParser()
from util import load_model_config, set_parameters

parser.add_argument("--data", type=str)
parser.add_argument("--tokenizer", type=str)

parser.add_argument("--n", type=int)
parser.add_argument("--embedding_dim")
parser.add_argument("--heads", type=int)
parser.add_argument("--d_ff", type=int)
parser.add_argument("--dropout_rate", type=float)
parser.add_argument("--eps", type=float)
parser.add_argument("--activation", type=str)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--mini_batch", type=int, default=10)
parser.add_argument("--shuffle_data", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--optimizer", type=str)
parser.add_argument('--early_stopping', type=float)

parser.add_argument("--pretrained_model", type=str)
parser.add_argument("--checkpoint", type=str)

args = parser.parse_args()

parameters = ['n', 'embedding_dim', 'heads', 'd_ff', 'activation', 'eps', 'dropout_rate', 'learning_rate', 'optimizer']


def program(data_folder: str, 
            tokenizer_path: str, 
            n: int, 
            embedding_dim: int, 
            heads: int, 
            d_ff: int, 
            dropout_rate: float, 
            eps: float, 
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]], 
            learning_rate: float, 
            batch_size: int, 
            epochs: int, 
            shuffle: bool, 
            mini_batch: int, 
            optimizer: optim.Optimizer, 
            checkpoint: str,
            pretrained_model_path: str = None):
    text_processor = TextProcessor(tokenizer_path=tokenizer_path)

    inputs = text_processor.load_data(f"{data_folder}/question.pkl")
    labels = text_processor.load_data(f"{data_folder}/answer.pkl")

    text_processor.loadd_tokenizer(tokenizer_path)

    token_size = text_processor.tokenizer.num_tokens + 1
    
    model = GPT(
        token_size=token_size,
        n=n,
        embedding_dim=embedding_dim,
        heads=heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        eps=eps,
        activation=activation,
        learning_rate=learning_rate, 
        checkpoint=checkpoint,
        optimizer=optimizer
    )

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)

    model.fit(inputs=inputs, labels=labels, pretrained_path=pretrained_model_path, batch_size=batch_size, epochs=epochs, mini_batch=mini_batch, shuffle_data=shuffle)

if __name__ == "__main__":
    if args.data is None or args.tokenizer is None or args.checkpoint is None:
        print("Missing Information")
    else:
        config = load_model_config(path='./config.yml')
        
        args = set_parameters(args, config['model_config'], parameters=parameters)
        print(args.checkpoint)
        if args.early_stopping is None:
            args.__dict__['early_stopping'] = config['early_stopping']['fine_tune']
        program(
            data_folder=args.data,
            tokenizer_path=args.tokenizer,
            n=args.n,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            d_ff=args.d_ff,
            dropout_rate=args.dropout_rate,
            eps=args.eps,
            activation=args.activation,
            learning_rate=args.learning_rate,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mini_batch=args.mini_batch,
            shuffle=args.shuffle_data,
            optimizer=args.optimizer,
            pretrained_model_path=args.pretrained_model
        )
