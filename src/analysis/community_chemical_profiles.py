"""
Chemical Characterization of Louvain Communities

For each of the 27 communities found in hiv_knn_graph.gexf, computes:
  - RDKit molecular descriptors (MW, LogP, TPSA, rings, HBD/HBA, RotBonds)
  - Functional group prevalences (35 SMARTS patterns)
  - HIV-activity enrichment (from labels)
  - Auto-generated chemical family name

Outputs:
  results/tables/community_chemical_profiles.csv   — per-community stats
  results/tables/community_chemical_report.txt     — readable named profiles
"""

import os, sys, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

JOINT_CSV      = "data/processed/joint_data.csv"
NODE_COMM_CSV  = "results/tables/louvain_node_communities.csv"
SCREENING_CSV  = "results/tables/brnpdb_consensus_screening.csv"
OUT_CSV        = "results/tables/community_chemical_profiles.csv"
OUT_TXT        = "results/tables/community_chemical_report.txt"
os.makedirs("results/tables", exist_ok=True)

# ── 35 SMARTS patterns (same as community_structural_patterns.py) ─
FUNCTIONAL_GROUPS = {
    "aromatic_benzene" : "c1ccccc1",
    "phenol"           : "c1ccc(O)cc1",
    "catechol"         : "c1cc(O)c(O)cc1",
    "carboxylic_acid"  : "C(=O)[OH]",
    "ester"            : "C(=O)O[C,c]",
    "amide"            : "C(=O)N",
    "primary_amine"    : "[NX3;H2][C,c]",
    "secondary_amine"  : "[NX3;H1]([C,c])[C,c]",
    "tertiary_amine"   : "[NX3;H0]([C,c])([C,c])[C,c]",
    "hydroxyl"         : "[CX4][OX2H]",
    "ether"            : "[OD2]([#6])[#6]",
    "ketone"           : "[#6][CX3](=O)[#6]",
    "aldehyde"         : "[CX3H1](=O)[#6]",
    "sulfate_ester"    : "OS(=O)(=O)O",
    "sulfonate"        : "S(=O)(=O)[O-,OH]",
    "phosphate"        : "OP(=O)(O)O",
    "nitro"            : "[NX3](=O)=O",
    "nitrile"          : "C#N",
    "thiol"            : "[#6][SX2H]",
    "thioether"        : "[#6][SX2][#6]",
    "halide_F"         : "[F]",
    "halide_Cl"        : "[Cl]",
    "halide_Br"        : "[Br]",
    "alkene"           : "C=C",
    "alkyne"           : "C#C",
    "pyridine"         : "n1ccccc1",
    "furan"            : "o1cccc1",
    "thiophene"        : "s1cccc1",
    "pyrrole"          : "[nH]1cccc1",
    "imidazole"        : "n1cncc1",
    "indole"           : "c1ccc2[nH]ccc2c1",
    "pyranose_sugar"   : "C1OC(O)C(O)C(O)C1O",
    "furanose_sugar"   : "C1OC(O)C(O)C1O",
    "lactone"          : "[#6][CX3](=O)O[#6]",
    "epoxide"          : "C1OC1",
}
FG_MOLS = {n: Chem.MolFromSmarts(s) for n, s in FUNCTIONAL_GROUPS.items()}


# ── Molecular properties ──────────────────────────────────────

def mol_props(mol):
    """Return dict of molecular descriptors. mol may be None → all NaN."""
    if mol is None:
        return {k: float("nan") for k in
                ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds",
                 "n_rings", "n_arom_rings", "n_heavy"]}
    return {
        "mw"         : Descriptors.MolWt(mol),
        "logp"       : Descriptors.MolLogP(mol),
        "tpsa"       : Descriptors.TPSA(mol),
        "hbd"        : Descriptors.NumHDonors(mol),
        "hba"        : Descriptors.NumHAcceptors(mol),
        "rotbonds"   : Descriptors.NumRotatableBonds(mol),
        "n_rings"    : rdMolDescriptors.CalcNumRings(mol),
        "n_arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_heavy"    : mol.GetNumHeavyAtoms(),
    }


def detect_fg(mol):
    """Return set of functional group names present in mol."""
    if mol is None:
        return set()
    return {n for n, p in FG_MOLS.items() if p is not None and mol.HasSubstructMatch(p)}


# ── Community naming heuristics ───────────────────────────────

def name_community(row):
    """
    Auto-assign a chemical family name based on RELATIVE enrichment of
    functional groups vs. global average, plus molecular descriptor deltas.

    `row` is a dict with:
      enr_*  : enrichment ratio (comm_frac / global_frac)
      fg_*   : absolute fraction
      d_*    : delta vs. global mean
      mean_* : absolute mean
    """
    enr = {k[4:]: v for k, v in row.items() if k.startswith("enr_")}
    fg  = {k[3:]: v for k, v in row.items() if k.startswith("fg_")}
    mw   = row.get("mean_mw", 0)
    logp = row.get("mean_logp", 0)
    rings = row.get("mean_n_rings", 0)
    arom  = row.get("mean_n_arom_rings", 0)
    hbd   = row.get("mean_hbd", 0)
    d_mw  = row.get("d_mw", 0)
    d_logp = row.get("d_logp", 0)
    d_rings = row.get("d_n_rings", 0)
    d_hbd   = row.get("d_hbd", 0)
    d_tpsa  = row.get("d_tpsa", 0)

    # --- Priority rules using RELATIVE enrichment (enr_X = comm/global) ---
    # An enrichment > 2x means this group is twice as common vs. the global avg.

    # Polyene macrolides: strongly enriched in alkene + lactone, high MW
    if enr.get("alkene", 1) > 2.5 and enr.get("lactone", 1) > 2.0 and d_mw > 50:
        return "Polyene macrolides / macrocyclic lactones"

    # Nucleosides: furanose_sugar + N-heterocycle enrichment
    if enr.get("furanose_sugar", 1) > 3.0 and (enr.get("pyridine", 1) > 3.0 or
                                                  enr.get("imidazole", 1) > 3.0):
        return "Nucleosides / nucleotides"

    # Aminoglycosides: pyranose_sugar + primary_amine enrichment
    if enr.get("pyranose_sugar", 1) > 3.0 and enr.get("primary_amine", 1) > 3.0:
        return "Aminoglycosides / sugar-amine conjugates"

    # Glycosides: pyranose_sugar enriched, large MW
    if enr.get("pyranose_sugar", 1) > 2.5 and d_mw > 30:
        return "Glycosides / saponins (sugar-decorated)"

    # Furanose / small sugars enriched
    if enr.get("furanose_sugar", 1) > 3.0 and d_mw < 20:
        return "Furanose-containing (small sugar-like)"

    # Polyphenols / tannins: catechol + high HBD enriched, high TPSA
    if enr.get("catechol", 1) > 2.5 and d_hbd > 1.0:
        return "Polyphenols / tannins (catechol-rich)"

    # Flavonoids: phenol + ketone enriched, moderate MW
    if enr.get("phenol", 1) > 2.0 and enr.get("ketone", 1) > 1.5 and d_hbd > 0.5:
        return "Flavonoids / isoflavonoids"

    # Hydroxycinnamic acids / phenylpropanoids: carboxylic_acid + alkene + benzene
    if enr.get("carboxylic_acid", 1) > 2.0 and enr.get("alkene", 1) > 1.5 and \
       enr.get("aromatic_benzene", 1) > 1.2 and d_mw < 0:
        return "Hydroxycinnamic acids / phenylpropanoids"

    # Fatty acids: carboxylic_acid enriched, low rings, positive logp delta
    if enr.get("carboxylic_acid", 1) > 2.0 and d_rings < -0.5 and d_logp > 0.5:
        return "Fatty acids / acyclic lipids"

    # Steroids / triterpenoids: high rings, low aromaticity, positive logp
    if d_rings > 0.3 and d_logp > 0.5 and enr.get("aromatic_benzene", 1) < 0.8:
        return "Steroids / triterpenoids (polycyclic)"

    # Sesquiterpene lactones: lactone enriched, moderate ring count
    if enr.get("lactone", 1) > 2.0 and 0 < d_rings < 0.5:
        return "Sesquiterpene lactones / terpenoid-lactones"

    # Halogenated: Cl or F enriched
    if enr.get("halide_Cl", 1) > 2.5 or enr.get("halide_F", 1) > 2.5:
        return "Halogenated compounds"

    # Nitrogen heterocycles: indole/pyrrole/imidazole enriched
    if enr.get("indole", 1) + enr.get("pyrrole", 1) + enr.get("imidazole", 1) > 5.0:
        return "Indole / pyrrole alkaloids"

    # Pyridine-rich: kinase inhibitor-like
    if enr.get("pyridine", 1) > 2.5:
        return "Pyridine-containing (kinase inhibitor-like)"

    # Large molecules, very polar (TPSA enriched, high HBD)
    if d_tpsa > 20 and d_hbd > 1.0:
        return "Large polar molecules (antibiotics / peptides)"

    # Epoxide-enriched
    if enr.get("epoxide", 1) > 3.0:
        return "Epoxide-bearing (reactive electrophiles)"

    # Sulfate / sulfonate
    if enr.get("sulfate_ester", 1) > 3.0 or enr.get("sulfonate", 1) > 3.0:
        return "Sulfated / sulfonated compounds"

    # High MW, low polarity → lipophilic drug-like
    if d_mw > 30 and d_logp > 0.3 and d_tpsa < 0:
        return "Lipophilic drug-like (high MW, low TPSA)"

    # Low MW, low polarity → fragment-like / small drug-like
    if d_mw < -20 and d_logp < 0:
        return "Small fragment-like molecules"

    # Low MW, high polarity
    if d_mw < -10 and d_tpsa > 10:
        return "Small polar molecules"

    # Default with most enriched group
    top_enr = max(enr.items(), key=lambda x: x[1]) if enr else (None, 1)
    if top_enr[1] > 2.0:
        return f"Enriched in {top_enr[0]} (x{top_enr[1]:.1f})"

    return "Drug-like / diverse (low chemical differentiation)"


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    joint      = pd.read_csv(JOINT_CSV)
    node_comm  = pd.read_csv(NODE_COMM_CSV)
    screening  = pd.read_csv(SCREENING_CSV)

    smiles_map = dict(zip(joint["joint_idx"], joint["smiles"]))
    pred_df    = pd.read_csv("results/tables/community_hiv_predictions.csv")
    # community → list of (brnpdb_id, name, score)
    comm_to_brnpdb = defaultdict(list)
    for _, r in pred_df.iterrows():
        comm_to_brnpdb[int(r["community"])].append(
            (int(r["brnpdb_id"]), str(r["common_name"]), float(r["activity_score"]))
        )

    print(f"  {len(joint):,} SMILES loaded")
    print(f"  {len(node_comm):,} nodes with community labels")
    print(f"  {len(node_comm['community'].unique())} communities")

    # ── Compute per-molecule descriptors ─────────────────────
    print("\nComputing RDKit descriptors for all molecules ...")
    t0 = time.time()

    props_list = []
    fg_list    = []
    for idx, row in node_comm.iterrows():
        nid  = int(row["node_id"])
        smi  = smiles_map.get(nid, None)
        mol  = None
        if smi and isinstance(smi, str):
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                pass
        props_list.append({**mol_props(mol), "node_id": nid,
                           "community": row["community"],
                           "is_brnpdb": row["is_brnpdb"],
                           "is_active": row["is_active"]})
        fg_list.append(detect_fg(mol))

        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1:,}/{len(node_comm):,} done ...")

    print(f"  Descriptors computed in {time.time()-t0:.1f}s")

    df = pd.DataFrame(props_list)
    df["fg"] = fg_list

    # ── Aggregate by community ────────────────────────────────
    print("\nAggregating by community ...")
    all_fg = list(FUNCTIONAL_GROUPS.keys())

    # Compute global fg prevalences (baseline for relative enrichment)
    all_fg_flat = Counter(g for fgs in df["fg"] for g in fgs)
    global_fg   = {g: all_fg_flat.get(g, 0) / len(df) for g in all_fg}
    global_rate = df[~df["is_brnpdb"]]["is_active"].mean()

    desc_cols = ["mw", "logp", "tpsa", "hbd", "hba",
                 "rotbonds", "n_rings", "n_arom_rings", "n_heavy"]
    global_means = {c: df[c].dropna().mean() for c in desc_cols}

    # BrNPDB node_id → brnpdb_id mapping from joint_data
    brnpdb_joint = joint[joint["brnpdb_id"].notna() & (joint["source"] == "Antiviral")]
    nodeid_to_brnpdbid = dict(zip(brnpdb_joint["joint_idx"].astype(int),
                                   brnpdb_joint["brnpdb_id"].astype(int)))
    brnpdb_names = {int(r["brnpdb_id"]): str(r["common_name"])
                    for _, r in screening.iterrows()}

    comm_rows = []

    for comm_id, sub in df.groupby("community"):
        n_total   = len(sub)
        n_hiv     = int((~sub["is_brnpdb"]).sum())
        n_brnpdb  = int(sub["is_brnpdb"].sum())
        n_active  = int(sub["is_active"].sum())
        enrich_rate = n_active / n_hiv if n_hiv > 0 else 0.0
        x_rate    = enrich_rate / global_rate if global_rate > 0 else 0.0

        # Numeric descriptor means vs. global means (delta)
        means  = {f"mean_{c}": sub[c].dropna().mean() for c in desc_cols}
        stds   = {f"std_{c}":  sub[c].dropna().std()  for c in desc_cols}
        deltas = {f"d_{c}": means[f"mean_{c}"] - global_means[c] for c in desc_cols}

        # Functional group prevalences + RELATIVE enrichment
        fg_counts = Counter(g for fgs in sub["fg"] for g in fgs)
        fg_fracs  = {g: fg_counts.get(g, 0) / n_total for g in all_fg}
        # Enrichment ratio: comm_frac / global_frac (clipped to avoid div/0)
        fg_enrich = {g: fg_fracs[g] / max(global_fg[g], 0.001) for g in all_fg}

        # Top ENRICHED groups (ratio > 1.3, frac > 1%) — what makes this community special
        top_enriched = sorted([(g, fg_enrich[g], fg_fracs[g])
                                for g in all_fg
                                if fg_enrich[g] > 1.3 and fg_fracs[g] > 0.01],
                               key=lambda x: -x[1])[:8]

        # Top DEPLETED groups (ratio < 0.6, global frac > 5%)
        top_depleted = sorted([(g, fg_enrich[g], fg_fracs[g])
                                for g in all_fg
                                if fg_enrich[g] < 0.6 and global_fg[g] > 0.05],
                               key=lambda x: x[1])[:4]

        # BrNPDB compound names in this community (from predictions CSV)
        brnpdb_items = comm_to_brnpdb.get(int(comm_id), [])
        brnpdb_info  = [f"{name[:28]} [{bid}]"
                        for bid, name, _ in sorted(brnpdb_items, key=lambda x: -x[2])]

        row_dict = {
            "community"         : int(comm_id),
            "n_total"           : n_total,
            "n_hiv"             : n_hiv,
            "n_brnpdb"          : n_brnpdb,
            "n_hiv_active"      : n_active,
            "hiv_enrich"        : round(enrich_rate, 4),
            "x_rate"            : round(x_rate, 2),
            "top_enriched_fg"   : "; ".join(f"{g}({r:.1f}x,{f:.0%})"
                                             for g, r, f in top_enriched),
            "top_depleted_fg"   : "; ".join(f"{g}({r:.2f}x)"
                                             for g, r, _ in top_depleted),
            "brnpdb_compounds"  : " | ".join(brnpdb_info[:5]),
            **{k: round(v, 3) for k, v in means.items()},
            **{k: round(v, 3) for k, v in stds.items()},
            **{k: round(v, 3) for k, v in deltas.items()},
            **{f"fg_{g}": round(v, 4) for g, v in fg_fracs.items()},
            **{f"enr_{g}": round(v, 3) for g, v in fg_enrich.items()},
        }
        comm_rows.append(row_dict)

    comm_df = pd.DataFrame(comm_rows).sort_values("x_rate", ascending=False)

    # Name each community
    comm_df["chemical_family"] = comm_df.apply(name_community, axis=1)

    comm_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # ── Readable report ───────────────────────────────────────
    global_rate = df[~df["is_brnpdb"]]["is_active"].mean()

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Chemical Profiles of Louvain Communities — hiv_knn_graph\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total communities  : {len(comm_df)}\n")
        f.write(f"Global HIV-active rate: {global_rate*100:.2f}%\n\n")

        for _, row in comm_df.iterrows():
            cid   = int(row["community"])
            n     = int(row["n_total"])
            enr   = float(row["hiv_enrich"])
            xr    = float(row["x_rate"])
            family = str(row["chemical_family"])

            f.write(f"Community {cid:>3}  [{family}]\n")
            f.write(f"  Size       : {n:,} nodes  "
                    f"({int(row['n_hiv'])} HIV + {int(row['n_brnpdb'])} BrNPDB)\n")
            f.write(f"  HIV active : {int(row['n_hiv_active'])}  "
                    f"({enr*100:.1f}%,  {xr:.1f}x global rate)\n")
            f.write(f"  MW         : {row['mean_mw']:.0f} ± {row['std_mw']:.0f} Da\n")
            f.write(f"  LogP       : {row['mean_logp']:.2f} ± {row['std_logp']:.2f}\n")
            f.write(f"  TPSA       : {row['mean_tpsa']:.0f} ± {row['std_tpsa']:.0f} A^2\n")
            f.write(f"  HBD / HBA  : {row['mean_hbd']:.1f} / {row['mean_hba']:.1f}\n")
            f.write(f"  Rings      : {row['mean_n_rings']:.1f} total  "
                    f"({row['mean_n_arom_rings']:.1f} aromatic)\n")
            f.write(f"  Top enriched: {str(row['top_enriched_fg'])}\n")
            f.write(f"  Depleted    : {str(row['top_depleted_fg'])}\n")
            if str(row.get('brnpdb_compounds', '')).strip():
                f.write(f"  BrNPDB      : {str(row['brnpdb_compounds'])[:200]}\n")
            f.write(f"  dMW={row['d_mw']:+.0f}  dLogP={row['d_logp']:+.2f}  "
                    f"dRings={row['d_n_rings']:+.2f}  dHBD={row['d_hbd']:+.2f}  "
                    f"dTPSA={row['d_tpsa']:+.0f}\n")
            f.write("\n")

    print(f"Saved: {OUT_TXT}")

    # ── Print summary table ───────────────────────────────────
    print("\n" + "=" * 90)
    print("COMMUNITY CHEMICAL PROFILES")
    print("=" * 90)
    print(f"{'C':>3}  {'Family':<44}  {'N':>5}  {'BR':>3}  "
          f"{'x':>5}  {'dMW':>5}  {'dLogP':>6}  {'dRing':>5}  {'dHBD':>5}  Top enriched groups")
    print("-" * 145)
    for _, row in comm_df.iterrows():
        enrich_str = "; ".join(p.split("(")[0] for p in
                                str(row["top_enriched_fg"]).split("; ")[:3])
        print(f"{int(row['community']):>3}  {str(row['chemical_family']):<44}  "
              f"{int(row['n_total']):>5}  {int(row['n_brnpdb']):>3}  "
              f"{row['x_rate']:>5.1f}x  {row['d_mw']:>+5.0f}  "
              f"{row['d_logp']:>+6.2f}  {row['d_n_rings']:>+5.2f}  "
              f"{row['d_hbd']:>+5.2f}  {enrich_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
