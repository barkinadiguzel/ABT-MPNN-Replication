from rdkit import Chem

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetHybridization().real,
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic())
    ]
