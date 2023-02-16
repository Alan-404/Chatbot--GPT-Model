#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_json("./datasets/conversation.json")
# %%
df.head(3)
# %%
df['conversation']
inputs = []
labels = []
# %%
for conversation in df['conversation']:
    for input in conversation['input']:
        for answer in conversation['answer']:
            inputs.append(input)
            labels.append(answer)
# %%
inputs
#%%
labels
# %%
from preprocessing.token_handle import TokenHandler
# %%
token_handler = TokenHandler()
# %%
inputs = token_handler.handle(inputs, type='input')
# %%
inputs
# %%
labels = token_handler.handle(labels, type='output')
# %%
labels
#%%
inputs
# %%
for i in range(len(inputs)):
    inputs[i] = inputs[i] + " " + labels[i]
# %%
inputs
# %%
labels
# %%
from preprocessing.text import TextProcessor
# %%
text_processor = TextProcessor("./conversation")
# %%
text_processor.fit(sequences=inputs)
# %%
text_processor.tokenizer.word_index
#%%
text_processor.tokenizer.num_index
# %%

# %%
inputs = text_processor.process(inputs, max_length=64)
# %%
labels = text_processor.process(labels, max_length=64)
# %%
inputs
# %%
token_size = len(text_processor.tokenizer.word_counts) + 1
# %%
token_size
# %%
from model.gpt import GPT
# %%
model = GPT(vocab_size=token_size, checkpoint='./saved_models/chatbot.pt')
# %%
import torch
# %%
train = torch.tensor(inputs)
#%%
target = torch.tensor(labels)
# %%
model.fit(train, batch_size=25, epochs=50)
#%%
model.save_model("./saved_models/chatbot.pt")
# %%
def answer(sequence: str):
    sequence = token_handler.handle([sequence], type='input')
    print(sequence)
    length = len(sequence[0].split(' '))
    sequence = text_processor.process(sequence, max_length=length-1)
    print(sequence)
    sequence = torch.tensor(sequence)
    result = model.predict(sequence=sequence, max_length=64, end_token=text_processor.tokenizer.word_index['__end__'])
    print(result)
    result = result[0][length:]
    answer = []
    for text in result:
        answer.append(text_processor.tokenizer.num_index[text.item()])
    return " ".join(answer)
# %%
answer("What is your name?")
# %%

# %%

# %%

# %%

# %%
