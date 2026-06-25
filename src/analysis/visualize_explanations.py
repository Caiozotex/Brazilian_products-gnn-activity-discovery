"""
Visualize GNNExplainer results.

For each BrNPDB compound, render the 2D molecule with:
  - Top-k important atoms highlighted (color-coded by importance)
  - Bonds between important atoms also highlighted
  - Title showing common_name, consensus_score, hiv_prob
  - Caption listing detected functional groups

Outputs PNG files into results/gnnexplainer/images/{tier}/
and one combined grid per community into images/communities/
"""

import os
import ast
import re
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings("ignore")

from src.models.gin_model import GINEEncoder
from src.models.task_models import HIVModel

CKPT             = "models/checkpoints/hiv/best_model.pt"
BRNPDB_DIR       = "data/processed/graphs_brnpdb_antiviral"
EXPLAN_CSV       = "results/gnnexplainer/brnpdb_explanations.csv"
STRUCT_CSV       = "results/gnnexplainer/brnpdb_structural_features.csv"
JOINT_CSV        = "data/processed/joint_data.csv"
OUT_BASE         = "results/gnnexplainer/images"
TOP_K            = 8
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for sub in ["high", "mid", "low", "communities"]:
    os.makedirs(os.path.join(OUT_BASE, sub), exist_ok=True)


def safe_filename(s, maxlen=60):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    return s[:maxlen].strip("_") or "unnamed"


# ── Re-load model + explainer to recompute per-atom importance ─
def load_model():
    encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128,
                          num_layers=4, dropout=0.2)
    model = HIVModel(encoder, hidden_dim=128)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, edge_attr, batch)


# ── Render one molecule with highlights ───────────────────────
def render_molecule(smiles, atom_importances, title, caption, out_path,
                    top_k=TOP_K, size=(520, 480)):
    """
    atom_importances : np.ndarray (N_atoms,) values in [0,1]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    n_atoms = mol.GetNumAtoms()
    importances = np.asarray(atom_importances[:n_atoms], dtype=float)

    # Pick top-k atoms
    if importances.max() > 0:
        importances = importances / importances.max()
    ranked = np.argsort(importances)[::-1]
    sel = ranked[:top_k].tolist()

    # Color by importance: red gradient
    atom_colors = {}
    for a in sel:
        w = float(importances[a])
        atom_colors[int(a)] = (1.0, 1.0 - 0.7 * w, 1.0 - 0.7 * w)

    # Bonds between selected atoms
    bond_indices = []
    bond_colors  = {}
    sel_set = set(sel)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in sel_set and b in sel_set:
            bond_indices.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = (1.0, 0.6, 0.6)

    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1] - 80)
    opts   = drawer.drawOptions()
    opts.addAtomIndices = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=sel,
        highlightAtomColors=atom_colors,
        highlightBonds=bond_indices,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    # Compose with title + caption
    mol_img = Image.open(__import__("io").BytesIO(png)).convert("RGB")
    canvas  = Image.new("RGB", size, "white")
    canvas.paste(mol_img, (0, 30))

    draw = ImageDraw.Draw(canvas)
    try:
        font_t = ImageFont.truetype("arial.ttf", 14)
        font_c = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_t = ImageFont.load_default()
        font_c = font_t

    title_safe   = title.encode("ascii", "replace").decode("ascii")
    caption_safe = caption.encode("ascii", "replace").decode("ascii")
    draw.text((6, 6), title_safe[:80], fill="black", font=font_t)

    y = size[1] - 48
    for line in caption_safe.split("\n")[:3]:
        draw.text((6, y), line[:90], fill="gray", font=font_c)
        y += 14

    canvas.save(out_path, "PNG")
    return True


def make_grid(image_paths, out_path, cols=3, tile=(520, 480), header=""):
    if not image_paths:
        return
    rows  = (len(image_paths) + cols - 1) // cols
    head_h = 36 if header else 0
    grid  = Image.new("RGB", (cols * tile[0], rows * tile[1] + head_h), "white")
    if header:
        draw = ImageDraw.Draw(grid)
        try:    font = ImageFont.truetype("arial.ttf", 18)
        except: font = ImageFont.load_default()
        draw.text((8, 8), header.encode("ascii", "replace").decode("ascii"),
                  fill="black", font=font)
    for i, p in enumerate(image_paths):
        r, c = divmod(i, cols)
        img  = Image.open(p).convert("RGB").resize(tile)
        grid.paste(img, (c * tile[0], head_h + r * tile[1]))
    grid.save(out_path, "PNG")


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    model   = load_model()
    wrapper = ModelWrapper(model)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    explan  = pd.read_csv(EXPLAN_CSV)
    struct  = pd.read_csv(STRUCT_CSV) if os.path.exists(STRUCT_CSV) else None
    joint   = pd.read_csv(JOINT_CSV)
    idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

    if struct is not None:
        explan = explan.merge(struct[["brnpdb_id", "functional_groups"]],
                              on="brnpdb_id", how="left", suffixes=("", "_s"))

    # Per-compound rendering ───────────────────────────────────
    tier_paths = {"high": [], "mid": [], "low": []}
    comm_paths = {}

    print(f"\nRendering {len(explan)} molecules ...")
    for _, row in explan.iterrows():
        brnpdb_id = int(row["brnpdb_id"])
        tier      = str(row.get("tier", "low"))
        community = row.get("community", None)
        consensus = float(row["consensus_score"])
        hiv_prob  = float(row["hiv_prob"])
        name      = str(row.get("common_name", ""))[:70]
        fgs       = str(row.get("functional_groups", row.get("functional_groups_s", "")))

        graph_path = os.path.join(BRNPDB_DIR, f"{brnpdb_id}.pt")
        if not os.path.exists(graph_path):
            continue

        graph = torch.load(graph_path, map_location=DEVICE, weights_only=False)
        x          = graph.x.to(DEVICE)
        edge_index = graph.edge_index.to(DEVICE)
        edge_attr  = graph.edge_attr.to(DEVICE) if graph.edge_attr is not None else None
        batch      = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)

        try:
            explanation = explainer(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                batch=batch, index=0,
            )
        except Exception as ex:
            print(f"  [skip] {brnpdb_id}: {ex}")
            continue

        nm = explanation.node_mask
        if nm is None:
            continue
        importances = nm.sum(dim=1).cpu().numpy() if nm.dim() == 2 else nm.cpu().numpy()

        # SMILES
        jidx_rows = joint[joint["brnpdb_id"] == brnpdb_id]
        if jidx_rows.empty:
            continue
        smi_raw = idx_to_smiles.get(int(jidx_rows["joint_idx"].iloc[0]), "")
        smiles  = str(smi_raw) if isinstance(smi_raw, str) else ""
        if not smiles:
            continue

        title   = f"{name}  [{tier.upper()}]"
        caption = (f"consensus={consensus:.3f} | hiv_prob={hiv_prob:.3f} | "
                   f"community={community}\n"
                   f"functional groups: {fgs[:90]}")
        fname   = f"{brnpdb_id}_{safe_filename(name, 40)}.png"
        out_path = os.path.join(OUT_BASE, tier, fname)

        ok = render_molecule(smiles, importances, title, caption, out_path)
        if ok:
            tier_paths.setdefault(tier, []).append(out_path)
            if community is not None and not (isinstance(community, float) and np.isnan(community)):
                comm_paths.setdefault(int(community), []).append((out_path, name, consensus))

        if (len(tier_paths["high"]) + len(tier_paths["mid"]) + len(tier_paths["low"])) % 20 == 0:
            done = sum(len(v) for v in tier_paths.values())
            print(f"  {done}/{len(explan)}")

    # Tier grids ────────────────────────────────────────────────
    for tier, paths in tier_paths.items():
        if not paths:
            continue
        out_grid = os.path.join(OUT_BASE, f"_grid_{tier}.png")
        make_grid(paths, out_grid, cols=3, header=f"BrNPDB compounds — {tier.upper()} consensus")
        print(f"Saved tier grid: {out_grid}  ({len(paths)} molecules)")

    # Community grids (only communities with >=2 BrNPDB) ───────
    for cid, items in comm_paths.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=lambda x: -x[2])
        paths = [p for p, _, _ in items_sorted]
        out_grid = os.path.join(OUT_BASE, "communities", f"community_{cid}.png")
        make_grid(paths, out_grid, cols=min(3, len(paths)),
                  header=f"Community {cid} — {len(paths)} BrNPDB compounds")
    print(f"Saved {sum(1 for v in comm_paths.values() if len(v) >= 2)} community grids "
          f"-> {os.path.join(OUT_BASE, 'communities')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
