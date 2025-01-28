import os
os.getcwd()

# wikihow corpus
txt = open('data/wikihow.txt', 'r').read()
print(len(txt)) # 621,684,876 characters
txt[0:100]

char_set = sorted(set(txt))
len(char_set) # 2,205 characters
char_set[-100:] # contains foreign script and emojis

# use tiktoken (BPE), cuz it is used for GPT
import tiktoken as tk

encoding = tk.get_encoding('o200k_base')
tokens = encoding.encode(txt)

# skip text cleaning for now

