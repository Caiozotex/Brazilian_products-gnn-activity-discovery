import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from typing import Optional


# -------------------- Atom features --------------------
def get_atom_features(atom: rdchem.Atom) -> torch.Tensor:
    """Return a 35-dimensional feature vector for an atom."""

    # ---- Atom type (10 + 1 "other") ----
    atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    atomic_num = atom.GetAtomicNum()

    atom_type_onehot = [1.0 if atomic_num == t else 0.0 for t in atom_types]
    if atomic_num not in atom_types:
        atom_type_onehot.append(1.0)  # other
    else:
        atom_type_onehot.append(0.0)

    # ---- Degree (0–5) ----
    degree = atom.GetDegree()
    degree_onehot = [0.0] * 6
    degree_onehot[min(degree, 5)] = 1.0

    # ---- Formal charge (-2 to +2 clipped) ----
    charge = max(-2, min(2, atom.GetFormalCharge()))
    charge_onehot = [0.0] * 5
    charge_onehot[charge + 2] = 1.0

    # ---- Number of hydrogens (0–4) ----
    num_h = atom.GetTotalNumHs()
    h_onehot = [0.0] * 5
    h_onehot[min(num_h, 4)] = 1.0

    # ---- Hybridization (5 + 1 "other") ----
    hyb_types = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]

    hyb = atom.GetHybridization()
    hyb_onehot = [1.0 if hyb == t else 0.0 for t in hyb_types]
    if hyb not in hyb_types:
        hyb_onehot.append(1.0)
    else:
        hyb_onehot.append(0.0)

    # ---- Aromaticity ----
    is_aromatic = [1.0 if atom.GetIsAromatic() else 0.0]

    # ---- Chirality ----
    is_chiral = [1.0 if atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED else 0.0]

    features = (
        atom_type_onehot
        + degree_onehot
        + charge_onehot
        + h_onehot
        + hyb_onehot
        + is_aromatic
        + is_chiral
    )

    return torch.tensor(features, dtype=torch.float)


# -------------------- Bond features --------------------
def get_bond_features(bond: rdchem.Bond) -> torch.Tensor:
    """Return a 9-dimensional feature vector for a bond."""

    # ---- Bond type ----
    type_onehot = [0.0] * 4
    bond_type = bond.GetBondType()

    if bond_type == rdchem.BondType.SINGLE:
        type_onehot[0] = 1.0
    elif bond_type == rdchem.BondType.DOUBLE:
        type_onehot[1] = 1.0
    elif bond_type == rdchem.BondType.TRIPLE:
        type_onehot[2] = 1.0
    elif bond_type == rdchem.BondType.AROMATIC:
        type_onehot[3] = 1.0
    else:
        type_onehot[0] = 1.0  # fallback

    # ---- Conjugation ----
    is_conj = [1.0 if bond.GetIsConjugated() else 0.0]

    # ---- Ring ----
    in_ring = [1.0 if bond.IsInRing() else 0.0]

    # ---- Stereo ----
    stereo_onehot = [0.0] * 3
    stereo = bond.GetStereo()

    if stereo == rdchem.BondStereo.STEREONONE:
        stereo_onehot[0] = 1.0
    elif stereo == rdchem.BondStereo.STEREOANY:
        stereo_onehot[1] = 1.0
    elif stereo in (rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE):
        stereo_onehot[2] = 1.0

    features = type_onehot + is_conj + in_ring + stereo_onehot

    return torch.tensor(features, dtype=torch.float)


# -------------------- SMILES to Mol --------------------
def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES to sanitized RDKit Mol."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except:
        return None

    return mol


# -------------------- Mol to PyG Data --------------------
def mol_to_graph(mol: Chem.Mol, y: Optional[float] = None):
    """Convert RDKit Mol to PyTorch Geometric Data object."""
    from torch_geometric.data import Data

    if mol is None:
        return None

    # ---- Node features ----
    x = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()], dim=0)

    # ---- Edges ----
    edges = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        feat = get_bond_features(bond)

        edges.append((i, j))
        edges.append((j, i))

        edge_features.append(feat)
        edge_features.append(feat.clone())

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 9), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


