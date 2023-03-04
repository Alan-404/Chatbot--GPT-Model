import torch
from argparse import ArgumentParser
from model.gpt import GPT
from preprocessing.text import TextProcessor, token_dictionary
import numpy as np
parser = ArgumentParser()
import re

parser.add_argument('--model', type=str)
parser.add_argument('--tokenizer', type=str)
parser.add_argument('--length', type=int, default=32)

parser.add_argument('--input', type=str)

args = parser.parse_args()


def program(model: str, tokenizer: str, input: str, max_len: int):
    text_processor = TextProcessor(tokenizer_path=tokenizer)

    text_processor.loadd_tokenizer(path=tokenizer)

    digits = text_processor.process(sequences=[input], max_len=None, start_token=True)
    digits = np.array(digits)
    digits = torch.tensor(digits)

    token_size = text_processor.tokenizer.num_tokens + 1

    gpt = GPT(token_size=token_size, checkpoint=model)

    result = gpt.predict(data=digits, limit_tokens=max_len, end_token=token_dictionary['end_token'])
    sequence = ""

    for item in result[0]:
        sequence += text_processor.tokenizer.index_token[item.item()] + " "
    return re.sub(input, "", sequence)


if __name__ == '__main__':
    if args.tokenizer is None or args.model is None or args.input is None:
        print("Missing data")
    else:
        print(program(
            model=args.model,
            tokenizer=args.tokenizer,
            max_len=args.length,
            input=args.input
        ))