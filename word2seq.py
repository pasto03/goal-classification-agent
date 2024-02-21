import string


class Word2Seq:
    def __init__(self) -> None:
        self.vocab = {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3}
        self.reverse_vocab = dict()

    def build_vocab(self, tokens):
        for t in tokens:
            if t not in self.vocab.keys():
                self.vocab[t] = len(self.vocab)

        self.reverse_vocab = {v:k for k, v in self.vocab.items()}

    def fit(self, tokens, max_len=10):
        seq = [self.vocab["PAD"]] * max_len
        for idx, t in enumerate(tokens[:max_len]):
            s = self.vocab[t] if t in self.vocab.keys() else self.vocab["UNK"]
            seq[idx] = s
        return seq
    
    def reverse_fit(self, sequence, no_pad=True):
        tokens = []
        for s in sequence:
            token = self.reverse_vocab[s] 
            if no_pad and token == 'PAD':
                break
            tokens.append(token)
        return tokens


if __name__ == "__main__":
    ws = Word2Seq()
    ws.build_vocab(list(string.ascii_lowercase))
    print(ws.vocab)
    print(ws.fit("abc"))
