import torch
from torch import nn

class TransformerTS(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.encoder(self.input(x))
        return self.fc(h[:,-1])
