import torch
import torch.nn as nn


class AtomAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, m, adj, dist, coulomb):
        bias = adj + dist + coulomb

        q = self.q(m)
        k = self.k(m)
        v = self.v(m)

        attn = (q @ k.transpose(-1, -2)) / (m.size(-1) ** 0.5)
        attn = attn + bias
        attn = torch.softmax(attn, dim=-1)

        return attn @ v + m
