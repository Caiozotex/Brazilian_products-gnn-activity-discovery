"""
GNNExplainer for BrNPDB antiviral compounds.

For each BrNPDB compound, runs GNNExplainer over the HIV model to find which
atoms and bonds drive the anti-HIV prediction. Results are grouped by consensus
score tier:
  - High  (consensus_score >= 0.5): model is confident about activity
  - Mid   (0.3 <= score < 0.5)
  - Low   (score < 0.3): model predicts inactive

Outputs
-------
results/gnnexplainer/
  brnpdb_explanations.csv   — per-compound: prediction, score, top atoms, fragment SMILES
  high_score_summary.txt    — readable report for high-confidence compounds
  low_score_summary.txt     — readable report for low-confidence compounds
"""

import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Batch

from src.models.gin_model import GINEEncoder, GraphPooling
from src.models.task_models import HIVModel

import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────
CKPT          = "models/checkpoints/hiv/best_model.pt"
BRNPDB_DIR    = "data/processed/graphs_brnpdb_antiviral"
SCREENING_CSV = "results/tables/brnpdb_consensus_screening.csv"
JOINT_CSV     = "data/processed/joint_data.csv"
OUT_DIR       = "results/gnnexplainer"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Atom feature index map (35-dim, same as rdkit_utils.py)
# First 11 dims = atom type one-hot → use atomic num from RDKit directly
ATOM_TYPES = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na',
              'Ca','Fe','As','Al','I','B','V','K','Tl','Yb','Sb',
              'Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge',
              'Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb']


# ── Load model ─────────────────────────────────────────────────
def load_model():
    encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128,
                          num_layers=4, dropout=0.2)
    model = HIVModel(encoder, hidden_dim=128)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ── Wrapper so GNNExplainer can call model(x, edge_index, edge_attr, batch) ──
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = self.model(x, edge_index, edge_attr, batch)
        return out  # (1, 1) logit


# ── Extract top-k important atoms from explanation ─────────────
def important_atoms(explanation, mol, top_k=None, threshold=0.5):
    """
    Returns list of atom indices ranked by importance.
    node_mask shape: (N_atoms,) or (N_atoms, F)
    """
    nm = explanation.node_mask
    if nm is None:
        return [], []
    if nm.dim() == 2:
        importance = nm.sum(dim=1).cpu().numpy()
    else:
        importance = nm.cpu().numpy()

    # Normalise
    if importance.max() > 0:
        importance = importance / importance.max()

    ranked = np.argsort(importance)[::-1]
    if top_k is not None:
        selected = ranked[:top_k].tolist()
    else:
        selected = ranked[importance[ranked] >= threshold].tolist()

    return selected, importance.tolist()


def mol_fragment_smiles(mol, atom_indices):
    """Return SMILES of the substructure formed by the given atom indices."""
    if not atom_indices:
        return ""
    try:
        # Include bonds between selected atoms
        bond_indices = []
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in atom_indices and \
               bond.GetEndAtomIdx() in atom_indices:
                bond_indices.append(bond.GetIdx())
        frag = Chem.MolFragmentToSmiles(mol, atom_indices, bond_indices)
        return frag
    except Exception:
        return ""


# ── Main ───────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # Load model and setup explainer
    model = load_model()
    wrapper = ModelWrapper(model)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="object",       # one scalar importance per atom
        edge_mask_type="object",       # one scalar importance per bond
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    # Load consensus scores
    screening = pd.read_csv(SCREENING_CSV)
    joint     = pd.read_csv(JOINT_CSV)
    idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

    records = []
    print(f"\nRunning GNNExplainer on {len(screening)} BrNPDB compounds ...\n")

    for _, row in screening.iterrows():
        brnpdb_id     = int(row["brnpdb_id"])
        common_name   = row["common_name"]
        consensus     = float(row["consensus_score"])
        community     = row.get("community", None)

        graph_path = os.path.join(BRNPDB_DIR, f"{brnpdb_id}.pt")
        if not os.path.exists(graph_path):
            print(f"  [skip] {brnpdb_id} — graph not found")
            continue

        graph = torch.load(graph_path, map_location=DEVICE, weights_only=False)
        x         = graph.x.to(DEVICE)
        edge_index = graph.edge_index.to(DEVICE)
        edge_attr  = graph.edge_attr.to(DEVICE) if graph.edge_attr is not None else None
        batch      = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)

        # Model prediction
        with torch.no_grad():
            logit = wrapper(x, edge_index, edge_attr, batch)
            prob  = torch.sigmoid(logit).item()

        # GNNExplainer
        try:
            explanation = explainer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                index=0,
            )
        except Exception as ex:
            print(f"  [error] {brnpdb_id}: {ex}")
            continue

        # Extract important atoms
        top_atoms, importances = important_atoms(explanation, mol=None, top_k=10)

        # Map top atoms to SMILES fragment
        # Find SMILES for this compound
        jidx  = joint[joint["brnpdb_id"] == brnpdb_id]["joint_idx"].values
        raw_smi = idx_to_smiles.get(int(jidx[0]), "") if len(jidx) > 0 else ""
        smiles  = str(raw_smi) if isinstance(raw_smi, str) else ""
        mol     = Chem.MolFromSmiles(smiles) if smiles else None
        frag_smiles = mol_fragment_smiles(mol, top_atoms) if mol else ""

        # Atom symbols for top-10
        atom_symbols = ""
        if mol:
            symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in top_atoms
                       if i < mol.GetNumAtoms()]
            atom_symbols = ",".join(symbols)

        tier = "high" if consensus >= 0.5 else ("mid" if consensus >= 0.3 else "low")

        records.append({
            "brnpdb_id":      brnpdb_id,
            "common_name":    common_name,
            "community":      community,
            "consensus_score": consensus,
            "tier":           tier,
            "hiv_prob":       round(prob, 4),
            "top_atoms_idx":  str(top_atoms),
            "top_atom_symbols": atom_symbols,
            "fragment_smiles": frag_smiles,
            "smiles":         smiles,
        })

        safe_name = common_name[:45].encode("ascii", "replace").decode("ascii")
        print(f"  [{tier:4}] {safe_name:<45}  "
              f"consensus={consensus:.3f}  hiv_prob={prob:.3f}  "
              f"top_atoms={atom_symbols}")

    # Save results
    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "brnpdb_explanations.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Readable summaries
    for tier, label in [("high", "HIGH consensus (>=0.5)"), ("low", "LOW consensus (<0.3)")]:
        sub = df[df["tier"] == tier].sort_values("consensus_score", ascending=(tier == "low"))
        path = os.path.join(OUT_DIR, f"{tier}_score_summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"GNNExplainer Summary — {label}\n")
            f.write("=" * 70 + "\n\n")
            for _, r in sub.iterrows():
                f.write(f"Compound  : {r['common_name']}\n")
                f.write(f"BrNPDB ID : {r['brnpdb_id']}\n")
                f.write(f"Community : {r['community']}\n")
                f.write(f"Consensus : {r['consensus_score']:.4f}\n")
                f.write(f"HIV prob  : {r['hiv_prob']:.4f}\n")
                f.write(f"Top atoms : {r['top_atom_symbols']}\n")
                f.write(f"Fragment  : {r['fragment_smiles']}\n")
                f.write(f"SMILES    : {r['smiles']}\n")
                f.write("-" * 70 + "\n\n")
        print(f"Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
