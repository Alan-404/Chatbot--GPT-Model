import torch
from argparse import ArgumentParser
from model.gpt import GPT
from preprocessing.text import TextProcessor, token_dictionary

parser = ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--tokenizer', type=str)
parser.add_argument('--length', type=int, default=32)

args = parser.parse_args()


def program(model: str, tokenizer: str, input: str, max_len: int):
    text_processor = TextProcessor(tokenizer_path=tokenizer)

    text_processor.loadd_tokenizer(path=tokenizer)

    digits = text_processor.process(sequences=[input], max_len=max_len, start_token=True)

    digits = torch.tensor(digits)

    token_size = text_processor.tokenizer.num_tokens + 1

    gpt = GPT(token_size=token_size, checkpoint=model)

    result = gpt.predict(data=digits, limit_tokens=max_len, end_token=token_dictionary['end_token'])

    return result
