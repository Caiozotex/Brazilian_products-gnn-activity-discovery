"""
Structural pattern mining across communities of BrNPDB compounds.

For every community that contains 2+ BrNPDB compounds we report:
  1. Murcko scaffold of each member  (the "skeleton" — ring systems + linkers)
  2. Maximum Common Substructure (MCS) across all members in the community
  3. Functional groups present in each member (SMARTS-based)
  4. Recurring functional groups in the community

This goes beyond atom-level explanations: it looks for *substructures*
(aromatic rings, sugars, sulfate groups, amides, etc.) that may explain
why these molecules cluster together and share a predicted activity profile.
"""

import os
import pandas as pd
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS
import warnings
warnings.filterwarnings("ignore")

SCREENING_CSV   = "results/tables/brnpdb_consensus_screening.csv"
EXPLANATIONS    = "results/gnnexplainer/brnpdb_explanations.csv"
JOINT_CSV       = "data/processed/joint_data.csv"
OUT_DIR         = "results/gnnexplainer"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Functional groups via SMARTS ──────────────────────────────
# Keys are human-readable names, values are SMARTS patterns.
FUNCTIONAL_GROUPS = {
    "aromatic_benzene"    : "c1ccccc1",
    "phenol"              : "c1ccc(O)cc1",
    "catechol"            : "c1cc(O)c(O)cc1",
    "carboxylic_acid"     : "C(=O)[OH]",
    "ester"               : "C(=O)O[C,c]",
    "amide"               : "C(=O)N",
    "primary_amine"       : "[NX3;H2][C,c]",
    "secondary_amine"     : "[NX3;H1]([C,c])[C,c]",
    "tertiary_amine"      : "[NX3;H0]([C,c])([C,c])[C,c]",
    "hydroxyl"            : "[CX4][OX2H]",
    "ether"               : "[OD2]([#6])[#6]",
    "ketone"              : "[#6][CX3](=O)[#6]",
    "aldehyde"            : "[CX3H1](=O)[#6]",
    "sulfate_ester"       : "OS(=O)(=O)O",
    "sulfonate"           : "S(=O)(=O)[O-,OH]",
    "phosphate"           : "OP(=O)(O)O",
    "nitro"               : "[NX3](=O)=O",
    "nitrile"             : "C#N",
    "thiol"               : "[#6][SX2H]",
    "thioether"           : "[#6][SX2][#6]",
    "halide_F"            : "[F]",
    "halide_Cl"           : "[Cl]",
    "halide_Br"           : "[Br]",
    "alkene"              : "C=C",
    "alkyne"              : "C#C",
    "pyridine"            : "n1ccccc1",
    "furan"               : "o1cccc1",
    "thiophene"           : "s1cccc1",
    "pyrrole"             : "[nH]1cccc1",
    "imidazole"           : "n1cncc1",
    "indole"              : "c1ccc2[nH]ccc2c1",
    "quinoline"           : "n1cccc2ccccc12",
    "pyranose_sugar"      : "C1OC(O)C(O)C(O)C1O",
    "furanose_sugar"      : "C1OC(O)C(O)C1O",
    "lactone"             : "[#6][CX3](=O)O[#6]",
    "epoxide"             : "C1OC1",
    "anhydride"           : "[CX3](=O)O[CX3](=O)",
    "guanidine"           : "NC(=N)N",
    "urea"                : "NC(=O)N",
    "steroid_4ring"       : "C1CCC2CCC3CCCC4CCCCC4(C3)C2C1",  # generic 6-6-6-5 fused
}

GROUP_MOLS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}


def detect_groups(mol):
    """Return list of functional group names present in mol."""
    found = []
    for name, patt in GROUP_MOLS.items():
        if patt is None:
            continue
        if mol.HasSubstructMatch(patt):
            found.append(name)
    return found


def murcko_scaffold_smiles(mol):
    """Return canonical SMILES of the Bemis-Murcko scaffold."""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        smi  = Chem.MolToSmiles(scaf)
        return smi if smi else "(none)"
    except Exception:
        return "(error)"


def compute_mcs(mol_list, timeout=10):
    """Return SMARTS of the maximum common substructure across mol_list."""
    if len(mol_list) < 2:
        return None
    try:
        mcs = rdFMCS.FindMCS(
            mol_list,
            timeout=timeout,
            completeRingsOnly=True,
            ringMatchesRingOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        return mcs.smartsString if mcs.numAtoms > 0 else None
    except Exception:
        return None


def ring_summary(mol):
    """Return a count of aromatic vs non-aromatic rings of each size."""
    ri = mol.GetRingInfo()
    rings = []
    for ring in ri.AtomRings():
        size = len(ring)
        is_arom = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        rings.append(f"{'arom' if is_arom else 'aliph'}{size}")
    return Counter(rings)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    explan   = pd.read_csv(EXPLANATIONS)
    joint    = pd.read_csv(JOINT_CSV)
    idx_to_smiles = joint.set_index("joint_idx")["smiles"].to_dict()

    # Ensure SMILES present
    if "smiles" not in explan.columns or explan["smiles"].isna().any():
        explan = explan.merge(joint[["brnpdb_id", "smiles"]], on="brnpdb_id",
                              how="left", suffixes=("", "_jx"))

    # ── Per-molecule structural info ──────────────────────────
    print("Computing per-molecule structural info ...")
    scaffolds, groups_list, rings_list = [], [], []
    mols = {}
    for _, row in explan.iterrows():
        smi = row.get("smiles", "")
        smi = str(smi) if isinstance(smi, str) else ""
        mol = Chem.MolFromSmiles(smi) if smi else None
        mols[int(row["brnpdb_id"])] = mol
        if mol is None:
            scaffolds.append("(invalid)")
            groups_list.append([])
            rings_list.append(Counter())
            continue
        scaffolds.append(murcko_scaffold_smiles(mol))
        groups_list.append(detect_groups(mol))
        rings_list.append(ring_summary(mol))

    explan["murcko_scaffold"]   = scaffolds
    explan["functional_groups"] = [";".join(g) for g in groups_list]
    explan["ring_summary"]      = [";".join(f"{k}x{v}" for k, v in r.items())
                                   for r in rings_list]

    # ── Per-compound CSV ──────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "brnpdb_structural_features.csv")
    explan.to_csv(out_csv, index=False)
    print(f"Saved per-compound features -> {out_csv}")

    # ── Community-level aggregation ───────────────────────────
    print("\nAggregating by community ...")
    community_rows = []
    sub_explan = explan.dropna(subset=["community"])
    sub_explan = sub_explan[sub_explan["community"].astype(str) != "nan"]
    sub_explan["community"] = sub_explan["community"].astype(int)

    for cid, sub in sub_explan.groupby("community"):
        if len(sub) < 2:
            continue
        ids = sub["brnpdb_id"].tolist()
        community_mols = [mols[int(i)] for i in ids if mols.get(int(i)) is not None]
        if len(community_mols) < 2:
            continue

        # MCS
        mcs_smarts = compute_mcs(community_mols)

        # Recurring functional groups (present in >50% of members)
        all_groups = []
        for i in ids:
            idx = sub_explan[sub_explan["brnpdb_id"] == i].index[0]
            all_groups.extend(groups_list[explan.index.get_loc(idx)])
        group_counts = Counter(all_groups)
        threshold    = max(2, int(0.5 * len(community_mols)))
        recurring    = {g: c for g, c in group_counts.items() if c >= threshold}

        # Shared Murcko scaffolds
        sub_scaffolds = [s for s in sub["murcko_scaffold"].tolist() if s and s != "(invalid)"]
        scaffold_counts = Counter(sub_scaffolds)
        shared_scaffolds = {s: c for s, c in scaffold_counts.items() if c >= 2 and s}

        # Mean consensus
        mean_consensus = float(sub["consensus_score"].mean())

        community_rows.append({
            "community":         int(cid),
            "n_brnpdb":          len(sub),
            "mean_consensus":    round(mean_consensus, 4),
            "compound_names":    " | ".join(sub["common_name"].astype(str).tolist())[:300],
            "mcs_smarts":        mcs_smarts or "(none)",
            "recurring_groups":  "; ".join(f"{g}({c})" for g, c in
                                            sorted(recurring.items(), key=lambda x: -x[1])),
            "shared_scaffolds":  "; ".join(f"{s} (x{c})" for s, c in
                                            sorted(shared_scaffolds.items(), key=lambda x: -x[1])),
        })

    comm_df = pd.DataFrame(community_rows).sort_values("n_brnpdb", ascending=False)
    out_comm = os.path.join(OUT_DIR, "community_structural_patterns.csv")
    comm_df.to_csv(out_comm, index=False)
    print(f"Saved community patterns -> {out_comm}")

    # ── Readable summary ──────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "community_patterns_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Structural Pattern Mining — BrNPDB Communities\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total communities with >=2 BrNPDB compounds: {len(comm_df)}\n\n")

        for _, r in comm_df.iterrows():
            f.write(f"Community {r['community']} | "
                    f"n={r['n_brnpdb']} | mean consensus={r['mean_consensus']:.3f}\n")
            f.write("-" * 75 + "\n")
            f.write(f"Compounds        : {r['compound_names']}\n")
            f.write(f"MCS (SMARTS)     : {r['mcs_smarts']}\n")
            f.write(f"Recurring groups : {r['recurring_groups'] or '(none)'}\n")
            f.write(f"Shared scaffolds : {r['shared_scaffolds'] or '(none)'}\n\n")

    print(f"Saved readable report   -> {report_path}")

    # ── Global functional-group summary ───────────────────────
    print("\nGlobal functional-group prevalence across all 147 BrNPDB compounds:")
    all_groups_flat = [g for lst in groups_list for g in lst]
    global_counts = Counter(all_groups_flat).most_common(15)
    for g, c in global_counts:
        print(f"  {g:<25} {c:>4} compounds ({c/len(explan)*100:.1f}%)")


if __name__ == "__main__":
    main()
