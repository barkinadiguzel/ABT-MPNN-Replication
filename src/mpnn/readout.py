import torch
import torch.nn as nn


class Readout(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, hidden_dim)
        )

    def forward(self, h_atoms):
        h_f = h_atoms.mean(dim=1)
        return self.ffn(h_f)
