#%%
import pandas as pd
import numpy as np
from model.gpt import GPT
from preprocessing.text import TextProcessor
from preprocessing.token_handle import TokenHandler
# %%
df = pd.read_json('./datasets/conversation.json')
# %%
df.head(10)
# %%
inputs = []
labels = []
# %%
for conversation in df['conversation']:
    for input in conversation['input']:
        for answer in conversation['answer']:
            inputs.append(input)
            labels.append(answer)
# %%
inputs
# %%
labels
# %%
sequences = inputs + labels
# %%
sequences
# %%
token_handler = TokenHandler()
# %%
sequences = token_handler.process(sequences=sequences)
# %%
sequences
# %%
text_processor = TextProcessor("./pretrain")
# %%
text_processor.fit(sequences=sequences)
# %%
text_processor.tokenizer.word_index
# %%
import torch
# %%
sequences = text_processor.process(sequences=sequences, max_length=64)
# %%
sequences
# %%
data = torch.tensor(sequences)
# %%
token_size = len(text_processor.tokenizer.word_index) + 1
# %%
gpt = GPT(vocab_size=token_size)
# %%
gpt.pretrain(sequences=data, batch_size=20, epochs=20)
# %%
gpt.save_model('./saved_models/pretrain.pt')
# %%
