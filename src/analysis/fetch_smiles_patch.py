"""Tenta resolver os 7 compostos restantes com buscas específicas."""
import csv, json, time, urllib.parse, urllib.request

INPUT_CSV = r"data/external/brnpdb_id.csv"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"
OPSIN   = "https://opsin.ch.cam.ac.uk/opsin/{}"

MANUAL_VARIANTS = {
    "7267": ["Acteoside", "Verbascoside",
             "beta-D-Glucopyranoside, 2-(3,4-dihydroxyphenyl)ethyl 3-O-(6-deoxy-alpha-L-mannopyranosyl)-, 4-[(2E)-3-(3,4-dihydroxyphenyl)-2-propenoate]"],
    "6940": ["beta-Amyrin", "beta-amyrin", "Olean-12-en-3beta-ol", "3beta-hydroxy-olean-12-ene"],
    "3887": ["4-nerolidylcatechol", "nerolidyl catechol", "4-(3,7,11-trimethyl-2,6,10-dodecatrienyl)catechol"],
    "6665": ["1-Naphthalenepropanoic acid, decahydro-5-(2-hydroxy-5-methylphenyl)-2-methylene-8a-methyl"],
    "5788": ["Germacrene D", "germacra-1(10),4,7(11)-triene", "Germacrene-D",
             "1,5-Cyclodecadiene, 1,5-dimethyl-8-methylidene"],
    "8749": ["Tachykinin peptide"],
    "9290": ["Convallatoxin", "Digitoxigenin", "Card-20(22)-enolide"],
}


def fetch(url_tmpl, name):
    try:
        url = url_tmpl.format(urllib.parse.quote(name))
        headers = {"Accept": "application/json" if "pubchem" in url else "chemical/x-daylight-smiles"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8").strip()
        if "pubchem" in url:
            j = json.loads(data)
            p = j["PropertyTable"]["Properties"][0]
            return p.get("IsomericSMILES") or p.get("SMILES") or ""
        else:
            return data if data and not data.startswith("Error") else ""
    except Exception:
        return ""

rows = []
with open(INPUT_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    for row in reader:
        rows.append(row)

missing = {r["brnpdb_id"]: r for r in rows if not r.get("smiles")}
print(f"Missing: {list(missing.keys())}")

for bid, r in missing.items():
    variants = MANUAL_VARIANTS.get(bid, [])
    smiles = ""
    for v in variants:
        smiles = fetch(PUBCHEM, v)
        if smiles:
            print(f"  [{bid}] PubChem OK: {v[:50]}")
            break
        time.sleep(0.2)
    if not smiles:
        smiles = fetch(OPSIN, r["common_name"])
        if smiles:
            print(f"  [{bid}] OPSIN OK")
    if not smiles:
        print(f"  [{bid}] STILL NOT FOUND")
    r["smiles"] = smiles
    time.sleep(0.2)

with open(INPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

total = sum(1 for r in rows if r.get("smiles"))
print(f"\nTotal: {total}/{len(rows)}")
