#%%
import torch
import numpy as np
import pandas as pd
import re
from constants.token import END_TOKEN, DELIM_TOKEN
# %%
df = pd.read_csv("./datasets/S08_question_answer_pairs.txt", sep='\t')
#%%
df = df.drop_duplicates(subset='Question')
df = df.dropna()
# %%
df['Content'] = df['ArticleTitle'] + " " + df['Question'] + " " + DELIM_TOKEN +" " + df['Answer'] + " " + END_TOKEN
# %%
df['Content']
# %%

# %%
X = np.array(df['Content'])
# %%
X[0]
# %%
from preprocessing.text import TextProcessor
# %%
text_handler = TextProcessor("./tokenizer")
# %%
max_length = 201
# %%
data = text_handler.process(sequences=X, max_length=max_length)
# %%
import pickle
# %%
with open('./clean/data.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%

# %%

# %%
# %%
