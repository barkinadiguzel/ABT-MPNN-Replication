import torch
import torch.nn as nn


class BondAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, m):
        q = self.query(m)
        k = self.key(m)
        v = self.value(m)

        attn = torch.softmax(q @ k.transpose(-1, -2) / (m.size(-1) ** 0.5), dim=-1)
        return attn @ v + m
