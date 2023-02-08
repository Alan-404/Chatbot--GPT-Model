#%%
from model.gpt import GPT
import torch
import pickle
import numpy as np
# %%
with open('./clean/inputs.pkl', 'rb') as handle:
    X_train = pickle.load(handle)
# %%
with open('./clean/labels.pkl', 'rb') as handle:
    y_train = pickle.load(handle)
# %%
with open('./tokenizer/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%
X_train.shape
# %%
y_train.shape
# %%
vocab_size = len(tokenizer.word_index) + 1
# %%
vocab_size
# %%
task_size = len(np.unique(y_train)) + 1
# %%
task_size
# %%
gpt = GPT(vocab_size=vocab_size, task_size=task_size)
# %%
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
#%%
gpt.fit(X_train, y_train, batch_size=7, epochs=5)
# %%
gpt.save_model("./saved_models/model")
# %%
