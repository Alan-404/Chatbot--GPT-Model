#%%
import re 
# %%
sequence = "he's"
# %%
pattern = r"(\w+)'s"
# %%
pronoun = ['he', 'she', 'it']
# %%
def handle(sequence: str):
    texts = sequence.split(' ')
    for i in range(len(texts)):
        check = re.findall(pattern, texts[i])
        if len(check) != 0 and check[0] in pronoun:
            texts[i] = re.sub(pattern, '\g<1> __BE__', texts[i])
        else:
            texts[i] = re.sub(pattern, '\g<1> __GENETIVE__', texts[i])
    return " ".join(texts)
# %%
handle("Tom's students")
# %%

# %%
result = re.findall(pattern, sequence)
# %%r
result[0]
# %%

# %%

# %%
