"""
Busca SMILES para cada composto em brnpdb_id.csv usando common_name.
Estratégia (em ordem):
  1. PubChem REST API por nome exato
  2. PubChem com variantes do nome (sem prefixo grego, sem sufixo de estereoquímica, CAS invertido)
  3. OPSIN (IUPAC→SMILES) para nomes sistemáticos
"""
import csv
import json
import re
import time
import urllib.parse
import urllib.request

INPUT_CSV  = r"data/external/brnpdb_id.csv"
OUTPUT_CSV = r"data/external/brnpdb_id.csv"

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"
OPSIN_URL   = "https://opsin.ch.cam.ac.uk/opsin/{}"

ALIAS_FRAGMENTS = {
    "L-Mannopyranose, 6-deoxy":  "L-Rhamnose",
    "D-Galactopyranose, 6-deoxy": "L-Fucose",
    "Stigmast-5-en-3-ol":         "Clionasterol",
    "fenil alcanoic":             "phenylacetic acid",
}

GREEK_PREFIX = re.compile(r'^[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]-')
STEREO_SUFFIX = re.compile(r',\s*\([^)]+\)-?\s*$')


def fetch_pubchem(name):
    try:
        url = PUBCHEM_URL.format(urllib.parse.quote(name))
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        return props.get("IsomericSMILES") or props.get("SMILES") or props.get("CanonicalSMILES") or ""
    except Exception:
        return ""


def fetch_opsin(name):
    try:
        url = OPSIN_URL.format(urllib.parse.quote(name))
        req = urllib.request.Request(url, headers={"Accept": "chemical/x-daylight-smiles"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            smiles = resp.read().decode("utf-8").strip()
            return smiles if smiles and not smiles.startswith("Error") else ""
    except Exception:
        return ""


def clean_name(name):
    c = GREEK_PREFIX.sub('', name)
    c = STEREO_SUFFIX.sub('', c).strip().rstrip(',').strip()
    return c


def cas_invert(name):
    parts = name.split(",", 1)
    if len(parts) == 2:
        return parts[1].strip() + " " + parts[0].strip()
    return name


def all_variants(name):
    variants = []
    for frag, alias in ALIAS_FRAGMENTS.items():
        if frag.lower() in name.lower() and alias:
            variants.append(alias)
    variants.append(name)
    c = clean_name(name)
    if c not in variants:
        variants.append(c)
    inv = cas_invert(name)
    if inv not in variants:
        variants.append(inv)
    inv_c = clean_name(inv)
    if inv_c not in variants:
        variants.append(inv_c)
    return variants


# Load CSV (reset smiles column)
rows = []
with open(INPUT_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    for row in reader:
        row["smiles"] = ""  # reset
        rows.append(row)

if "smiles" not in fieldnames:
    fieldnames.append("smiles")

print(f"Processing {len(rows)} compounds...\n")

found_count = 0
for i, r in enumerate(rows, 1):
    bid  = r["brnpdb_id"]
    name = r["common_name"]
    safe = name[:60].encode("ascii", "replace").decode()
    smiles = ""

    # 1. PubChem variants
    for v in all_variants(name):
        smiles = fetch_pubchem(v)
        if smiles:
            break
        time.sleep(0.15)

    # 2. OPSIN fallback
    if not smiles:
        smiles = fetch_opsin(clean_name(name))
        if smiles:
            time.sleep(0.2)

    r["smiles"] = smiles
    status = "OK " if smiles else "---"
    print(f"[{i:3d}/{len(rows)}] {status} [{bid}] {safe}")
    if smiles:
        found_count += 1
    time.sleep(0.2)

# Save
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. {found_count}/{len(rows)} compounds have SMILES.")
print(f"Saved to {OUTPUT_CSV}")
