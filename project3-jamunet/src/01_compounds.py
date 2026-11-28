"""
01_compounds.py — Mengambil struktur kimia senyawa jamu dari PubChem.

Alur:
1. Baca daftar kurasi manual  data/plant_compounds_curated.csv
   (tanaman -> senyawa yang DILAPORKAN di literatur, lengkap dengan sumbernya).
2. Untuk setiap nama senyawa unik, tanya PubChem PUG-REST:
   "nama ini -> CID berapa? SMILES-nya apa? rumus molekulnya? InChIKey?"
3. Lakukan hal yang sama untuk obat antidiabetes pembanding
   (dipakai nanti di 07_similarity_ml.py).
4. Simpan tabel hasil ke results/tables/.

PubChem PUG-REST dokumentasi: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
Contoh URL:
https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/curcumin/property/SMILES,InChIKey/JSON
"""

from urllib.parse import quote

import pandas as pd

from common import DATA_DIR, TABLE_DIR, cached_request

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# Properti yang diminta. Catatan: sejak 2025 PubChem mengganti nama kolom
# "IsomericSMILES" -> "SMILES" dan "CanonicalSMILES" -> "ConnectivitySMILES".
# Kita minta keduanya supaya script tetap jalan di versi lama maupun baru.
PROPS = "MolecularFormula,MolecularWeight,SMILES,IsomericSMILES,CanonicalSMILES,InChIKey,IUPACName"


def fetch_pubchem(name: str) -> dict:
    """Cari satu senyawa di PubChem berdasarkan namanya."""
    url = f"{PUG}/compound/name/{quote(name)}/property/{PROPS}/JSON"
    resp = cached_request("pubchem", url)
    if resp is None:                      # 404 -> nama tidak dikenal PubChem
        return {"query_name": name, "found": False}
    # Satu nama bisa cocok dengan beberapa CID; ambil yang pertama
    # (PubChem mengurutkan berdasarkan relevansi, biasanya CID "utama").
    p = resp["PropertyTable"]["Properties"][0]
    smiles = p.get("SMILES") or p.get("IsomericSMILES")          # dengan stereokimia
    return {
        "query_name": name,
        "found": True,
        "cid": int(p["CID"]),
        "formula": p.get("MolecularFormula"),
        "mw_pubchem": float(p.get("MolecularWeight", "nan")),
        "smiles": smiles,
        "inchikey": p.get("InChIKey"),
        "iupac_name": p.get("IUPACName"),
    }


def main():
    # ---------------- 1. Senyawa jamu ----------------
    curated = pd.read_csv(DATA_DIR / "plant_compounds_curated.csv")
    print(f"Daftar kurasi: {len(curated)} pasangan tanaman-senyawa, "
          f"{curated['plant_id'].nunique()} tanaman, "
          f"{curated['compound'].nunique()} senyawa unik")

    rows = [fetch_pubchem(n) for n in sorted(curated["compound"].unique())]
    comp = pd.DataFrame(rows)
    not_found = comp.loc[~comp["found"], "query_name"].tolist()
    if not_found:
        print("  [!] Tidak ditemukan di PubChem:", not_found)
    comp = comp[comp["found"]].drop(columns="found")

    # Satu senyawa bisa berasal dari >1 tanaman (mis. eugenol: kayu manis & salam)
    plants = (curated.groupby("compound")["plant_id"]
              .apply(lambda s: "; ".join(sorted(set(s)))).rename("plants"))
    klass = curated.drop_duplicates("compound").set_index("compound")["compound_class"]
    comp = comp.join(plants, on="query_name").join(klass, on="query_name")
    comp = comp.rename(columns={"query_name": "compound"})

    # Tabel panjang tanaman-senyawa (dengan sumber) + CID, untuk dashboard
    plant_table = curated.merge(comp[["compound", "cid", "smiles"]],
                                on="compound", how="inner")

    comp.to_csv(TABLE_DIR / "01_compounds_pubchem.csv", index=False)
    plant_table.to_csv(TABLE_DIR / "01_plant_compound_table.csv", index=False)
    print(f"  -> {len(comp)} senyawa berhasil diambil dari PubChem")

    # ---------------- 2. Obat antidiabetes pembanding ----------------
    drugs = pd.read_csv(DATA_DIR / "antidiabetic_drugs.csv")
    drows = [fetch_pubchem(n) for n in drugs["drug"]]
    dtab = pd.DataFrame(drows).rename(columns={"query_name": "drug"})
    dtab = drugs.merge(dtab[dtab["found"]].drop(columns="found"), on="drug")
    dtab.to_csv(TABLE_DIR / "01_antidiabetic_drugs_pubchem.csv", index=False)
    print(f"  -> {len(dtab)} obat antidiabetes pembanding diambil dari PubChem")
    print(comp[["compound", "cid", "formula", "plants"]].to_string(index=False))


if __name__ == "__main__":
    main()
