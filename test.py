#%%
from preprocessing.text import TextProcessor
# %%
sequences = ["She loves going to New York city.", 'hello, world']
# %%
text_processor = TextProcessor("test/test.pkl")
# %%
train = text_processor.process(sequences=sequences, max_len=20, start_token=True, end_token=True)
# %%
train

# %%

# %%
text_processor.tokenizer.token_index
# %%
import spacy
# %%
text = "Tom loves going to New York city"
# %%
nlp = spacy.load('en_core_web_sm')
# %%
doc = nlp(text)
for token in doc:
    print(token.tag_)
    print(token.lemma_)
# %%

# %%
