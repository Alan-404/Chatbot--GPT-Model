#%%
import torch
import spacy
import numpy as np
import pandas as pd
# %%
from preprocessing.text import TextProcessor
# %%
df = pd.read_json('./datasets/train.json')
# %%
df.head(10)
# %%
df.columns
# %%
data = list()
# %%
for index, row in df.iterrows():
    data.append(row['question'])
    for answer in row['nq_answer']:
        data.append(answer)
# %%
len(data)
# %%
text_processor = TextProcessor('./pretrain/pretrain.pkl')
# %%
data
# %%
train = text_processor.process(sequences=data, max_len=64, start_token=True, end_token=True)
# %%
train
# %%
train.shape
# %%
import pickle
# %%
with open('./clean/pretrain_data.pkl', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
from model.gpt import GPT
# %%
with open('./clean/pretrain_data.pkl', 'rb') as handle:
    train = pickle.load(handle)
# %%
with open('./pretrain/pretrain.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%

# %%

# %%
token_size = tokenizer.num_tokens + 1
# %%
token_size
# %%

# %%
gpt = GPT(vocab_size=token_size)
#%%
import torch
# %%
inputs = torch.tensor(train)
# %%

# %%
gpt.fit(inputs=inputs, batch_size=32, epochs=5, shuffle_data=True, mini_batch=10)
# %%
