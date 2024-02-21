# utils.py
import re
import spacy
import pickle
from word2seq import Word2Seq
import torch

ws = Word2Seq()
ws = pickle.load(open("ws_en.pkl", "rb"))
src_vocab_size = len(ws.vocab) + 1

MAX_LEN_U = 11

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Model will be using {} device.".format(device))

nlp = spacy.load("en_core_web_sm")
def tokenize(data):
    return [[token.text for token in doc] for doc in list(nlp.pipe(data))]

def extract_text(text):
    text = re.sub("[.,:?!]", "", text.lower())
    return text

def encode_input(q: str):
    q = extract_text(q)
    q = ws.fit(tokenize([q.lower()])[0], max_len=MAX_LEN_U)
    q = torch.tensor(q).unsqueeze(0)

    return q

