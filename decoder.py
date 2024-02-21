import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, p):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, self.num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, output_size)

        return predictions, hidden, cell
    

if __name__ == "__main__":
    from encoder import Encoder
    batch_size = 16
    seq_len = 10
    src_vocab_size = 20
    tgt_vocab_size = 24
    embedding_dim = 100
    hidden_size = 1024
    p = 0.1
    i = torch.randint(0, src_vocab_size, size=(seq_len, batch_size))
    encoder = Encoder(input_size=src_vocab_size, embedding_size=embedding_dim, hidden_size=hidden_size,
                      p=p)
    encoder_states, e_hidden, e_cell = encoder(i)
    # print(hidden.shape)   # [num_layers, batch_size, hidden_size]

    decoder = Decoder(input_size=src_vocab_size, embedding_size=embedding_dim, hidden_size=hidden_size,
                      output_size=tgt_vocab_size, p=p)
    
    t = 0
    out_t, h_t, c_t = decoder(i[t], encoder_states, e_hidden, e_cell)
    print(out_t.shape, h_t.shape)
