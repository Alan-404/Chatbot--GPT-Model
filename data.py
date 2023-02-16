#%%
import pandas as pd
from constants.token import Token
import re
token = Token()
import pickle
from preprocessing.text import TextProcessor
# %%
df = pd.read_json('./datasets/train.json')
# %%
df.head(10)
# %%
df['annotations'][0][0]['qaPairs']
#%%
type(df['annotations'][0][0]['qaPairs'])
# %%
X = []
y = []
sentences = []
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
    sequence = f"{token.START_TOKEN} " + change_token(row['question']) + f" {token.DELIM_TOKEN}"
    answer = answer + f" {token.END_TOKEN}"
    sentence = sequence + " " + answer
    X.append(sequence)
    y.append(answer)
    sentences.append(sentence)
    """ if 'qaPairs' in row['annotations'][0]:
        annotations = row['annotations'][0]['qaPairs']
    else:
        continue
    for qa in annotations:
        seq = f"{token.START_TOKEN} " + change_token(qa['question']) + f" {token.DELIM_TOKEN} "  + change_token(qa['answer'][0])+ f" {token.END_TOKEN}"
        X.append(seq)
        y.append(row['viewed_doc_titles'][0]) """
    
# %%
X
#%%
len(X)
# %%
y
# %%
len(X)
# %%
len(y)
#%%
len(sentences)
# %%
import numpy as np
# %%
text_handler = TextProcessor(tokenizer_path='./tokenizer')
# %%
text_handler.fit(sequences=X+y)
# %%
X_train = text_handler.process(X, max_length=64)
y_train = text_handler.process(y, max_length=64)
# %%
X_train
# %%
y_train
# %%
text_handler.tokenizer.word_index
# %%
X
# %%
text_handler.tokenizer.word_index['__delim__']
# %%
with open('./clean/inputs.pkl', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
with open('./clean/outputs.pkl', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
X+y
# %%
len(X+y)
# %%
