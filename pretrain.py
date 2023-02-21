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
#%%
df['len'] = df['question'].apply(lambda x: len(x.split(' ')))
# %%

# %%
df['len'].max()
# %%

# %%

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
train = text_processor.process(sequences=data, max_len=32, start_token=True, end_token=True)
# %%
text_processor.save_data(train, path="./clean", filename="pretrain_data.pkl")
# %%
train = text_processor.load_data("./clean/pretrain_data.pkl")
# %%
train.shape
# %%
text_processor.loadd_tokenizer("./pretrain/pretrain.pkl")
# %%
text_processor.tokenizer.token_index
# %%
import pickle
with open("./clean/pretrain_data.pkl", 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%

# %%

# %%
from model.gpt import GPT
# %%

# %%

# %%
token_size = text_processor.tokenizer.num_tokens + 1
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
inputs.size()
# %%
inputs.device
# %%

# %%
gpt.pretrain(data=inputs, batch_size=64, epochs=5, shuffle_data=True, mini_batch=10)
# %%
gpt.save_model(path='./saved_models/pretrain.pt')
# %%
