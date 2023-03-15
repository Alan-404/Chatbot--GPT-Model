from model.gpt import GPT
from preprocessing.text import TextProcessor
import numpy as np
import torch
class Predictor:
    def __init__(self, checkpoint: str, tokenizer: str, limit_token: int) -> None:
        self.checkpoint = checkpoint
        self.text_processor = TextProcessor(tokenizer_path=tokenizer)
        self.text_processor.loadd_tokenizer(tokenizer)
        token_size = self.text_processor.tokenizer.num_tokens + 1
        self.max_len = limit_token
        self.model = GPT(token_size=token_size, checkpoint=checkpoint)
        self.end_token = self.text_processor.tokenizer.token_index['__end__']
    def predict(self, input: str):
        digits = self.text_processor.process(inputs=[input], outputs=[], max_len=None, start_token=True)
        digits = np.array(digits)
        digits = torch.tensor(digits)        

        result = self.model.predict(seq=digits, num_tokens=self.max_len, end_token=self.end_token)
        sequence = ""

        for item in result[0][digits.size(1):]:
            sequence += self.text_processor.tokenizer.index_token[item.item()] + " "
        return sequence