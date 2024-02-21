import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab_size, teacher_force_ratio=0.5, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.tgt_vocab_size = tgt_vocab_size
        self.teacher_force_ratio = teacher_force_ratio
        
    def forward(self, source, target):
        batch_size = source.shape[1]
        target_len = target.shape[0]

        outputs = torch.zeros(target_len, batch_size, self.tgt_vocab_size).to(self.device)

        encoder_states, hidden, cell = self.encoder(source)

        # first input to decoder is <sos> token
        x = target[0]

        for t in range(1, target_len):
            # use encoder hidden and cell as decoder start hidden and cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # store next output prediction
            outputs[t] = output

            # output: [batch_size, tgt_vocab_size]
            best_guess = output.argmax(1)

            # teacher forcing 
            x = target[t] if random.random() < self.teacher_force_ratio else best_guess

        return outputs