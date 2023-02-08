#%%
import pandas as pd
# %%
df = pd.read_json('./datasets/train.json')
# %%
df.head(10)
# %%
import json
# %%
df['annotations'][0][0]['qaPairs']
#%%
type(df['annotations'][0][0]['qaPairs'])
# %%
X = []
y = []
# %%
from constants.token import Token
import re
token = Token()
# %%
def change_token(sequence: str):
    sequence = re.sub("\u2026", f" {token.TRIPLE_DOT_TOKEN}", sequence)
    sequence = re.sub(r"(\b[A-Z])\.(?=[A-Z]\b|\s|$)", f" {token.SEP_TOKEN}", sequence)
    sequence = re.sub('[?]', f" {token.QUESTION_TOKEN}", sequence)
    sequence = re.sub(",", f" {token.COMMA_TOKEN}", sequence)
    sequence = re.sub("\n", f" {token.LINE_TOKEN}", sequence)
    return sequence

#%%
def change_list(sequences: list):
    result = []
    for item in sequences:
        result.append(change_token(item))

    return result
# %%
for index, row in df.iterrows():
    answer = f" {token.OR_TOKEN} ".join(change_list(sequences=row['nq_answer']))
    sequence = f"{token.START_TOKEN} " + change_token(row['question']) + f" {token.DELIM_TOKEN} " + answer + f" {token.END_TOKEN}"
    X.append(sequence)
    y.append(row['viewed_doc_titles'][0])
    if 'qaPairs' in row['annotations'][0]:
        annotations = row['annotations'][0]['qaPairs']
    else:
        continue
    for qa in annotations:
        seq = change_token(qa['question']) + f" {token.DELIM_TOKEN} " + change_token(qa['answer'][0])+ f" {token.END_TOKEN}"
        X.append(seq)
        y.append(row['viewed_doc_titles'][0])
    
# %%
X
# %%
y
# %%
len(X)
# %%
len(y)
# %%
import numpy as np
# %%
np.unique(y)
# %%
label_dict = dict()
# %%
count = 1
for item in y:
    if item not in label_dict:
        label_dict[item] = count
        count += 1
# %%
label_dict
# %%
import pickle
# %%
with open('./clean/labels_dictionary.pkl', 'wb') as handle:
    pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
y_train = []
for item in y:
    y_train.append(label_dict[item])
# %%
y_train
# %%
y_train = np.array(y_train)
# %%
y_train.shape
# %%
with open('./clean/labels.pkl', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
max_length = 171
# %%
from preprocessing.text import TextProcessor
# %%
text_handler = TextProcessor(tokenizer_path='./tokenizer')
# %%
x_train = text_handler.process(sequences=X, max_length=max_length)
# %%
x_train.shape
# %%
with open('./clean/inputs.pkl', 'wb') as handle:
    pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%