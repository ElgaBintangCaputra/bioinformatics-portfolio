"""
03_targets.py
=============
Langkah 3: Mengambil target protein manusia teruji eksperimental dari ChEMBL.

Bagaimana cara kerja bioinformatika di balik layar?
1. Setiap senyawa jamu dicari ID ChEMBL-nya berdasarkan InChIKey (sidik jari struktur
   kimia yang unik secara internasional, jauh lebih akurat daripada nama!).
2. Dari ID ChEMBL senyawa tersebut, kita tarik seluruh data bioaktivitas pada target
   manusia (*Homo sapiens*, tipe protein tunggal SINGLE PROTEIN) dengan nilai
   pChEMBL >= 5.5 (setara IC50/Ki/Kd <= ~3.16 uM — ambang standar farmakologi produk alam).
3. Target ChEMBL ID diterjemahkan menjadi simbol gen HGNC resmi (mis. `PTGS2`, `AKT1`, `PPARG`)
   agar bisa digabungkan dengan Open Targets dan jaringan protein STRING!

Semua respons API disimpan ke cache lokal data/cache/chembl/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import FIG_DIR, TABLE_DIR, cached_request

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PCHEMBL_MIN = 5.5


def cari_chembl_id(inchikey: str, name: str) -> str | None:
    """Cari molecule_chembl_id dari InChIKey, fallback ke nama senyawa."""
    if inchikey:
        url = f"{CHEMBL_API}/molecule.json"
        resp = cached_request(
            "chembl", url,
            params={"molecule_structures__standard_inchi_key": inchikey, "format": "json"},
        )
        mols = (resp or {}).get("molecules", [])
        if mols:
            return mols[0]["molecule_chembl_id"]

    resp = cached_request(
        "chembl", f"{CHEMBL_API}/molecule.json",
        params={"pref_name__iexact": name, "format": "json"},
    )
    mols = (resp or {}).get("molecules", [])
    return mols[0]["molecule_chembl_id"] if mols else None


def ambil_aktivitas_manusia(chembl_id: str) -> list[dict]:
    """Ambil bioaktivitas manusia dengan pChEMBL >= PCHEMBL_MIN (dengan pagination)."""
    url = f"{CHEMBL_API}/activity.json"
    activities = []
    offset = 0
    while offset < 400:  # batas aman per senyawa
        resp = cached_request(
            "chembl", url,
            params={
                "molecule_chembl_id": chembl_id,
                "target_organism": "Homo sapiens",
                "pchembl_value__gte": PCHEMBL_MIN,
                "limit": 100,
                "offset": offset,
                "format": "json",
            },
        )
        if not resp:
            break
        batch = resp.get("activities", [])
        activities.extend(batch)
        if not resp.get("page_meta", {}).get("next"):
            break
        offset += 100
    return activities


def ambil_simbol_target_batch(target_ids: list[str]) -> dict[str, tuple[str | None, str, str]]:
    """Petakan daftar target_chembl_id secara batch (hemat waktu API ChEMBL 50x lipat!)."""
    out = {}
    batch_size = 40
    for i in range(0, len(target_ids), batch_size):
        chunk = sorted(target_ids[i : i + batch_size])
        resp = cached_request(
            "chembl",
            f"{CHEMBL_API}/target.json",
            params={"target_chembl_id__in": ",".join(chunk), "limit": batch_size, "format": "json"},
        )
        for t in (resp or {}).get("targets", []):
            tid = t.get("target_chembl_id")
            ttype = t.get("target_type", "")
            pname = t.get("pref_name", "")
            sym = None
            if ttype == "SINGLE PROTEIN":
                for comp in t.get("target_components", []):
                    for syn in comp.get("target_component_synonyms", []):
                        if syn.get("syn_type") == "GENE_SYMBOL":
                            sym = syn.get("component_synonym")
                            break
                    if sym:
                        break
            out[tid] = (sym, pname, ttype)
    return out


def main():
    print(f"== Langkah 3: Penelusuran Target Protein Manusia di ChEMBL (pChEMBL >= {PCHEMBL_MIN}) ==")
    comp = pd.read_csv(TABLE_DIR / "01_compounds_pubchem.csv")

    raw_acts = []
    matched_compounds = 0

    for _, r in comp.iterrows():
        cname = r["compound"]
        cid_chembl = cari_chembl_id(r.get("inchikey", ""), cname)
        if not cid_chembl:
            print(f"  [-] {cname}: tidak ada di ChEMBL")
            continue
        matched_compounds += 1
        acts = ambil_aktivitas_manusia(cid_chembl)
        print(f"  [+] {cname} ({cid_chembl}): {len(acts)} catatan bioaktivitas", flush=True)
        for a in acts:
            tid = a.get("target_chembl_id")
            if tid:
                raw_acts.append((cname, r["plants"], cid_chembl, tid, a.get("standard_type", ""), float(a["pchembl_value"])))

    unique_tids = sorted({x[3] for x in raw_acts})
    print(f"  Mengambil anotasi gen untuk {len(unique_tids)} target unik secara batch...", flush=True)
    target_cache = ambil_simbol_target_batch(unique_tids)

    raw_links = []
    for cname, plants, cid_chembl, tid, stype, pval in raw_acts:
        gene_sym, tname, ttype = target_cache.get(tid, (None, "", ""))
        if not gene_sym:
            continue
        raw_links.append({
            "compound": cname,
            "plants": plants,
            "molecule_chembl_id": cid_chembl,
            "target_chembl_id": tid,
            "gene_symbol": gene_sym,
            "target_name": tname,
            "standard_type": stype,
            "pchembl_value": pval,
        })

    df = pd.DataFrame(raw_links)
    if df.empty:
        raise RuntimeError("Tidak ada target ditemukan dari ChEMBL!")

    # Gabungkan duplikat senyawa-gen dengan mengambil pChEMBL maksimum (aktivitas terkuat)
    agg = (
        df.groupby(["compound", "plants", "molecule_chembl_id", "target_chembl_id", "gene_symbol", "target_name"], as_index=False)
        .agg(max_pchembl=("pchembl_value", "max"), n_assays=("pchembl_value", "count"))
        .sort_values(["max_pchembl", "n_assays"], ascending=[False, False])
    )
    agg.to_csv(TABLE_DIR / "03_compound_targets.csv", index=False)

    print(f"  Senyawa cocok di ChEMBL: {matched_compounds}/{len(comp)}")
    print(f"  Total interaksi unik senyawa-gen: {len(agg)}")
    print(f"  Jumlah senyawa yang punya target manusia: {agg['compound'].nunique()}")
    print(f"  Jumlah gen target manusia unik: {agg['gene_symbol'].nunique()}")

    # Plot jumlah target unik per senyawa
    counts = agg.groupby("compound")["gene_symbol"].nunique().sort_values().tail(20)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    counts.plot(kind="barh", color="#1f77b4", ax=ax, alpha=0.9)
    ax.set_title(f"Jumlah Target Protein Manusia per Senyawa Jamu (ChEMBL pChEMBL >= {PCHEMBL_MIN})", fontweight="bold")
    ax.set_xlabel("Jumlah Gen Target Unik")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_targets_per_compound.png", dpi=220)
    plt.close()
    print("Selesai Langkah 3.")


if __name__ == "__main__":
    main()
