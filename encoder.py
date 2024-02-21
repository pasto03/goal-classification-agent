import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1    # must be 1 to adapt to any seq_len

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, self.num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size*2)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:self.num_layers], hidden[self.num_layers:self.num_layers*2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:self.num_layers], cell[self.num_layers:self.num_layers*2]), dim=2))

        return encoder_states, hidden, cell
    

if __name__ == "__main__":
    batch_size = 16
    src_vocab_size = 20
    seq_len = 10
    embedding_dim = 100
    hidden_size = 1024
    p = 0.1
    i = torch.randint(0, src_vocab_size-1, size=(seq_len, batch_size))
    encoder = Encoder(input_size=src_vocab_size, embedding_size=embedding_dim, hidden_size=hidden_size,
                      p=p)
    encoder_states, hidden, cell = encoder(i)
    print(encoder_states.shape)
    print(hidden.shape)   # [num_layers, batch_size, hidden_size]

