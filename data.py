#%%
import spacy
# %%
nlp = spacy.load('en_core_web_sm')
# %%
text = "I #comma hello"
# %%
doc = nlp(text)
for token in doc:
    print(token)
# %%
