import numpy as np

# these are for the placeholder ofc

def build_matrices(num_atoms):
    adj = np.eye(num_atoms)
    dist = np.zeros((num_atoms, num_atoms))
    coulomb = np.zeros((num_atoms, num_atoms))

    return adj, dist, coulomb
