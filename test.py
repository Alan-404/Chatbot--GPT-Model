#%%
import pickle
# %%
with open('./clean/pretrain_data.pkl', 'rb') as handle:
    data = pickle.load(handle)
# %%
with open('./clean/question.pkl', 'rb') as handle:
    q = pickle.load(handle)
# %%
with open('./clean/answer.pkl', 'rb') as handle:
    a = pickle.load(handle)
# %%
import numpy as np
# %%
np.unique(data)
# %%
np.unique(a)
# %%
np.unique(q)
# %%
