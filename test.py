#%%
import torch
# %%
a = torch.rand((10, 10))
# %%
a.device
# %%
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# %%
a.to(device)
# %%
a.device
# %%
