import torch
import torch.nn as nn

from .message_functions import MessageFunction
from .bond_attention import BondAttention
from .atom_attention import AtomAttention
from .readout import Readout


class ABTMPNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim, T=3):
        super().__init__()

        self.T = T

        self.message_fn = MessageFunction(atom_dim, bond_dim, hidden_dim)
        self.bond_attn = BondAttention(hidden_dim)
        self.atom_attn = AtomAttention(hidden_dim)
        self.readout = Readout(hidden_dim, hidden_dim * 2)

        self.atom_proj = nn.Linear(atom_dim, hidden_dim)

    def forward(self, atom_feat, bond_feat, adj, dist, coulomb):

        h = self.atom_proj(atom_feat)

        # Message passing iterations
        for _ in range(self.T):
            m = self.message_fn(h, bond_feat)
            m = self.bond_attn(m)
            h = self.atom_attn(m, adj, dist, coulomb)

        return self.readout(h)
