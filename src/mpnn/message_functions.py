import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageFunction(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(atom_dim + bond_dim, hidden_dim)

    def forward(self, h_v, e_vw):
        x = torch.cat([h_v, e_vw], dim=-1)
        return F.relu(self.linear(x))
