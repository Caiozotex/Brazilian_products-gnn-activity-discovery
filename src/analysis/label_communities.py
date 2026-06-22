import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

# Full chemical family characterization of 27 Louvain communities
# Based on inspection of BrNPDB compound names + RDKit descriptor deltas + FG enrichment
communities_labeled = [
    (11, 13.0,  1, "Macrolida polienica (Amphotericin B)",
     "Anfotericina B: macrolida com cadeia polienica C38 e anel lactona. Enriquecimento em tioeter (1.3x) e F (1.3x) no HIV space, tipico de inibidores aprovados."),
    (17,  2.3,  7, "Fitoesterois C28-C29 + acidos cinnamicos prenilados",
     "Estigmasterol (3 estereoisomeros) + ergostano sulfatado + acidos 3-[4-OH-3,5-bis(prenilfenil)]propenoicos. Esqueleto ciclopentanoperidrofenantrenico com cadeia lateral longa."),
    (16,  2.1,  1, "Acido cafeoico (hidroxicinamato catecol)",
     "Acido 3,4-dihidroxicinnamico. Catecol + acido propenoico. Comunidade HIV levemente enriquecida em aldeidos e sulfonatos."),
    (5,   2.0,  0, "Zona HIV drug-like glicosidica e sulfonada (sem BrNPDB)",
     "Sem representantes BrNPDB. Enriquecida em piranose (1.7x), OH (1.5x) e furano (1.4x): possivelmente inibidores com farmacofor glicosidico."),
    (24,  1.1,  1, "Alcaloide bisindolico macrociclico",
     "Ciclooct[diindol]-dicarboxilato: anel ciclooctadieno fusionado a dois nucleos indolicos. Classe isolada de esponjas marinhas com atividade antiviral reportada."),
    (9,   0.85, 2, "Flavonois simples (quercetina e derivados)",
     "Quercetina e 3-O-metilquercetina: flavonois 3-OH com catecol no anel B."),
    (2,   0.8, 35, "Acidos graxos C10-C18 + triterpenos pentaciclicos + sesquiterpenos biciclicos",
     "Maior cluster BrNPDB (35 compostos). Mistura: acidos graxos lineares (C10-C18), acidos ursolicop/oleanoico, sesquiterpenos (nerolidol, sabineno, pineno), peptideo ciclico disulfeto, aldeidos graxos."),
    (6,   0.6, 21, "Sesquiterpenos ciclopropazulenoides (cedrano/gorgonano) + cardenolideo",
     "16 compostos 1H-cicloprop[e]azuleno decahidro-1,1,4/7-trimetil: sesquiterpenos triciclicos com ciclopropano. Tambem glicosideo cardenolidico, xantona glucosilada, fucose e galactose livres."),
    (7,   0.6,  1, "Flavonol O-metilado (caempferida)",
     "Caempferida = 4-O-metilkaempferol. Comunidade HIV enriquecida em imidazol (1.3x)."),
    (3,   0.6, 13, "Flavonas arabinosiladas + catecois sesquiterpenoides",
     "Luteolina, luteolina-arabinosideo, catecol nerolidilico, acido naftalenopropanoico, mirceno, linalil acetato. Perfil fenolico + terpenoide misto."),
    (4,   0.5,  8, "Flavonois poliglicosilados + glicosideo fenilpropanoide (verbascoside)",
     "Verbascoside + quercetina-3-O-rhamnosideo + rutina + flavonois allopyranosideo. Todos glicosilados com ligacao O-glicosidica."),
    (13,  0.5,  9, "Pentoses livres + sesquiterpenos macrociclicos endoperoxidicos",
     "D-Ribose, D/L-Arabinose, D-Xilose, L-Lixose (pentoses). Mais cicloundecatrieno e oxabiciclos com epoxido endoperoxidico."),
    (12,  0.5,  7, "Hexoses e acidos uronicoslivres",
     "L-Fucose, L-Ramnose, D-Glucose, D-Galactose, D-Manose, ac. D-Galacturonico, ac. D-Glucuronico. Monossacarideos polares. HIV dataset enriquecido em pirrol/indol mas BrNPDB contribuem apenas acucares."),
    (19,  0.5,  0, "Zona HIV drug-like com aldeidos (sem BrNPDB)",
     "Sem representantes BrNPDB. Leve enriquecimento em aldeidos (1.6x)."),
    (20,  0.4,  1, "Flavona C-glucosilada (vitexina/orientina)",
     "4H-1-Benzopiran-4-ona 6-beta-D-glucopiranosil-5,7-diOH: C-glicosideo com ligacao C-C direta ao anel flavonico."),
    (8,   0.4,  1, "Acido graxo omega-aromatico",
     "Acido omega-fenilalcanoico: cadeia alifatica terminada em benzeno. Intermediario entre lipidio e fenilpropanoide."),
    (15,  0.3,  2, "p-Cumarico + benzoato de benzila (fenilpropanoide + ester aromatico)",
     "Acido p-cumarico (4-OH-cinnamico) e benzoato de benzila (ester diaromatico). Fenilpropanoides sem catecol."),
    (21,  0.3,  1, "Diidroflavonol (taxifolina)",
     "2,3-diidro-3,5,7-triOH-flavanona: taxifolina. Anel C parcialmente saturado C2-C3."),
    (1,   0.3,  9, "Monoterpenos monocíclicos e oxidados (mentano, cineol)",
     "Limoneno, terpinoleno, alpha/gamma-terpineno, 1,8-cineol, biciclo-undecadieno. Todos C10, volateis, tipicos de oleos essenciais. Inclui acido malico."),
    (22,  0.3,  2, "Acido ferulico + flavona O-metilada (hispidulina)",
     "Acido ferulico (metoxicinnamato) e hispidulina (flavona 5,6,7-triOH-3-metoxi). Fenilpropanoide metilado + flavona metilada."),
    (0,   0.3,  0, "Zona HIV drug-like generica (sem BrNPDB)",
     "Sem representantes BrNPDB. Sem caracteristica estrutural distinta vs. media global."),
    (26,  0.3,  0, "Zona HIV drug-like rica em tiois (sem BrNPDB)",
     "Sem representantes BrNPDB. Enriquecida em tiol (1.6x): possivelmente peptidomimeticos ou compostos covalentes anti-HIV."),
    (18,  0.3,  0, "Zona HIV drug-like rica em indol (sem BrNPDB)",
     "Sem representantes BrNPDB. Enriquecida em indol (1.4x): similar a inibidores de integrase."),
    (25,  0.2, 25, "Sesquiterpenos macrociclicos (cariofilano/humulano) + lactonas poliacetilenicas + aciclovir",
     "Cariofileno/humulano (5 compostos C15), butenolidas poliacetilenicas com cadeia octatriyn-1-il (7 compostos), monoterpenos (pineno, camfeno, linalol), aciclovir (antiviral clinico!), naftalenometanol sesquiterpenico."),
    (14,  0.2,  0, "Zona HIV drug-like generica (sem BrNPDB)",
     "Sem representantes BrNPDB. Sem diferenciacao estrutural significativa."),
    (10,  0.2,  0, "Zona HIV drug-like com alcinos e piranoses (sem BrNPDB)",
     "Sem representantes BrNPDB. Enriquecida em piranose (1.8x), alcino (1.6x) e sulfonato (1.4x): possivelmente inibidores tipo stavudina/NRTIs com farmacofor alcino."),
    (23,  0.1,  0, "Zona HIV drug-like de menor MW (sem BrNPDB)",
     "Sem representantes BrNPDB. delta MW=-12 Da vs. media global: compartimento de compostos menores no espaco quimico."),
]

rows = []
for comm_id, xrate, n_br, family, reasoning in communities_labeled:
    rows.append({
        "comunidade": comm_id,
        "familia_quimica": family,
        "n_brnpdb": n_br,
        "taxa_hiv_x": xrate,
        "explicacao": reasoning,
    })

df = pd.DataFrame(rows).sort_values("taxa_hiv_x", ascending=False)
df.to_csv("results/tables/community_chemical_families.csv", index=False, encoding="utf-8")
print("Saved: results/tables/community_chemical_families.csv")
print(f"Total: {len(df)} communities")

# Print for display
print()
print(f"{'C':>3}  {'x':>5}  {'BR':>3}  Família")
print("-" * 90)
for _, row in df.iterrows():
    print(f"{row['comunidade']:>3}  {row['taxa_hiv_x']:>5.1f}x  {row['n_brnpdb']:>3}  {row['familia_quimica']}")
