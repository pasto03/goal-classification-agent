from torch import nn
from encoder import Encoder
from utils import src_vocab_size

class CommandClassifier(nn.Module):
    def __init__(self, input_len, output_dim, hidden_size=64, embedding_dim=250, p=0.1):
        super().__init__()
        self.encoder = Encoder(input_size=src_vocab_size, embedding_size=embedding_dim, 
                               hidden_size=hidden_size, p=p)   # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(input_len * hidden_size * 2, 32),   # [input_len * hidden_size * 2, 32]
            nn.ReLU(),
            nn.Linear(32, output_dim)   # [32, goal_space]
        )
        
    def forward(self, x):
        out, _, _ = self.encoder(x)
        out = self.classifier(out.view(out.shape[0], -1))
        return out
    