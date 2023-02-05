#%%
import torch
from model.gpt import GPT
import numpy as np
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# %%
with open('./tokenizer/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%
a = torch.tensor(np.array([[1,5,6]]))
# %%
a.shape
# %%
vocab_size = len(tokenizer.word_counts) + 1
# %%
vocab_size
# %%
model = GPT(vocab_size=vocab_size, checkpoint='./saved_models/05_02_14h25_gpt')
# %%
result = model.predict(a)
# %%
result
# %%

# %%

# %%

# %%
