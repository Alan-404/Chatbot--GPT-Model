#%%
from model.gpt import GPT
import torch
import pickle
# %%
with open('./tokenizer/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# %%
vocab_size = len(tokenizer.word_index)+1
# %%
model = GPT(vocab_size=vocab_size, checkpoint='./saved_models/05_02_14h25_gpt')
# %%
with open('./clean/data.pkl', 'rb') as handle:
    data = pickle.load(handle)
# %%
data.shape
#%%
data = torch.tensor(data)
# %%
model.fit(sequences=data, batch_size=15, epochs=5)
# %%
model.save_model("./saved_models/05_02_14h25_gpt")
# %%

# %%

# %%
