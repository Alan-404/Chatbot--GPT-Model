import torch
import torch.nn.functional as F
from model.gpt import GPT
from argparse import ArgumentParser
from typing import Union, Callable
from preprocessing.text import TextProcessor

parser = ArgumentParser()

parser.add_argument("--data", type=str)
parser.add_argument("--tokenizer", type=str)

parser.add_argument("--n", type=int, default=12)
parser.add_argument("--embedding_dim", type=int, default=768)
parser.add_argument("--heads", type=int, default=12)
parser.add_argument("--d_ff", type=int, default=2048)
parser.add_argument("--dropout_rate", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=1e-7)
parser.add_argument("--activation", type=Union[str, Callable[[torch.Tensor], torch.Tensor]], default=F.relu)
parser.add_argument("--learning_rate", type=float, default=0.0007)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--mini_batch", type=int, default=10)
parser.add_argument("--shuffle_data", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=1)

parser.add_argument("--checkpoint", type=str)

args = parser.parse_args()



def program(data_path: str, tokenizer_path: str, n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]], learning_rate: float, checkpoint: str, epochs: int, batch_size: int, mini_batch: int, shuffle: bool):
    text_processor = TextProcessor(tokenizer_path=tokenizer_path)

    dataset = text_processor.load_data(data_path)

    dataset = torch.tensor(dataset)

    text_processor.loadd_tokenizer(tokenizer_path)

    token_size = text_processor.tokenizer.num_tokens + 1
    
    gpt = GPT(token_size=token_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation, learning_rate=learning_rate, checkpoint=checkpoint)
    
    gpt.pretrain(data=dataset, epochs=epochs, batch_size=batch_size, shuffle_data=shuffle, mini_batch=mini_batch)


if __name__ == "__main__":
    if args.data is None or args.tokenizer is None or args.checkpoint is None:
        print("Missing Data")
    else:
        program(
            data_path=args.data,
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
            shuffle=args.shuffle_data
        )
    

    
