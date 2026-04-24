import os
import torch
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


# ------------------------------------------------------------
# Mapping from atom type one‑hot index to element symbol
# (as defined in rdkit_utils.get_atom_features)
ATOM_TYPES = [
    'H',   # 0 → Hydrogen
    'C',   # 1 → Carbon
    'N',   # 2 → Nitrogen
    'O',   # 3 → Oxygen
    'F',   # 4 → Fluorine
    'P',   # 5 → Phosphorus
    'S',   # 6 → Sulfur
    'Cl',  # 7 → Chlorine
    'Br',  # 8 → Bromine
    'I',   # 9 → Iodine
    'X'    # 10 → Other (fallback)
]

# Color mapping for Gephi (hex codes or RGB)
# Gephi accepts colors as #RRGGBB or <r g b> in 0‑1 range
ATOM_COLORS = {
    'H':  '#FFFFFF',   # white
    'C':  '#A0A0A0',   # gray
    'N':  '#3050F0',   # blue
    'O':  '#F04040',   # red
    'F':  '#90E090',   # light green
    'P':  '#FFA500',   # orange
    'S':  '#E0C080',   # tan
    'Cl': '#1F8A1F',   # dark green
    'Br': '#A52A2A',   # brown
    'I':  '#9932CC',   # purple
    'X':  '#000000'    # black for others
}

# Bond type decoding: [single, double, triple, aromatic]
BOND_TYPES = ['single', 'double', 'triple', 'aromatic']

# ------------------------------------------------------------
def decode_atom_features(node_features: torch.Tensor) -> Dict[str, Any]:
    """
    Decode a 40‑dim atom feature tensor into human‑readable properties.
    Feature layout (see rdkit_utils.get_atom_features):
      [0:10]  atom type one‑hot (H, C, N, O, F, P, S, Cl, Br, I)
      [10:16] degree one‑hot (0..5)
      [16:21] formal charge one‑hot (-2..+2)
      [21:26] number of hydrogens one‑hot (0..4)
      [26:31] hybridization one‑hot (sp, sp2, sp3, sp3d, sp3d2)
      [31]    aromatic (0/1)
      [32]    chiral (0/1)
    """
    # Atom type
    atom_onehot = node_features[0:10]
    atom_idx = torch.argmax(atom_onehot).item()
    if atom_onehot[atom_idx] == 0:   # fallback if all zeros
        atom_idx = 10
    atom_symbol = ATOM_TYPES[atom_idx]

    # Degree (number of bonded neighbours)
    deg_onehot = node_features[10:16]
    degree = torch.argmax(deg_onehot).item()
    # Note: if degree == 5, that means '5 or more'

    # Aromaticity
    aromatic = bool(node_features[31].item() > 0.5)

    # (Optional) hybridization and charge could also be extracted, but not required for basic viz
    return {
        'atom': atom_symbol,
        'degree': degree,
        'aromatic': 1 if aromatic else 0,
        'color': ATOM_COLORS.get(atom_symbol, '#000000')
    }

def decode_bond_features(bond_attr: torch.Tensor) -> Dict[str, Any]:
    """
    Decode a 7‑dim bond feature tensor.
    Layout: [0:4] bond type (single, double, triple, aromatic)
            [4]   conjugated (0/1)
            [5]   in_ring (0/1)
            [6:9] stereo (ignored for Gephi)
    """
    bond_type_onehot = bond_attr[0:4]
    bt_idx = torch.argmax(bond_type_onehot).item()
    bond_type = BOND_TYPES[bt_idx] if bt_idx < len(BOND_TYPES) else 'single'
    conjugated = bool(bond_attr[4].item() > 0.5)

    return {
        'bond_type': bond_type,
        'conjugated': 1 if conjugated else 0
    }

def pytorch_geo_to_networkx(data) -> nx.Graph:
    """
    Convert a PyTorch Geometric Data object (with x, edge_index, edge_attr)
    into a NetworkX graph with node/edge attributes.
    """
    G = nx.Graph()
    num_nodes = data.num_nodes

    # Add nodes
    for i in range(num_nodes):
        node_feat = data.x[i]          # shape (40,)
        attrs = decode_atom_features(node_feat)
        G.add_node(i, **attrs)

    # Add edges (avoid duplicates because G is undirected)
    edge_index = data.edge_index       # shape (2, E)
    edge_attr = data.edge_attr         # shape (E, 7)
    for e_idx in range(edge_index.shape[1]):
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        # Only add each edge once (src < dst) because NetworkX will treat as undirected
        if src < dst:
            bond_attrs = decode_bond_features(edge_attr[e_idx])
            G.add_edge(src, dst, **bond_attrs)
    return G

# ------------------------------------------------------------
def convert_pt_files_in_directory(input_dir: str, output_base: str):
    """
    Convert all .pt files in input_dir into:
    output_base/<input_folder_name>/*.gexf
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Create output folder with SAME name as input folder
    output_path = Path(output_base) / input_path.name
    output_path.mkdir(parents=True, exist_ok=True)

    pt_files = list(input_path.glob("*.pt"))
    if not pt_files:
        print(f"⚠️ No .pt files found in {input_dir}")
        return

    print(f"📂 Converting from: {input_dir}")
    print(f"📁 Saving to: {output_path}\n")

    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location='cpu')

            G = pytorch_geo_to_networkx(data)

            base_name = pt_file.stem
            G.graph['name'] = base_name

            gexf_file = output_path / f"{base_name}.gexf"
            nx.write_gexf(G, str(gexf_file))

            print(f"✓ {pt_file.name} -> {gexf_file.name}")

        except Exception as e:
            print(f"✗ Failed {pt_file.name}: {e}")


# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt graphs to GEXF")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .pt graph files"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="gexf_output",
        help="Base directory for GEXF output"
    )

    args = parser.parse_args()

    convert_pt_files_in_directory(
        input_dir=args.input_dir,
        output_base=args.output_base
    )