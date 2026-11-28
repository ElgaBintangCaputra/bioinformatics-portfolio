"""
07_similarity_ml.py
===================
Langkah 7: Ruang Kimia (Morgan Fingerprints), Kemiripan Tanimoto ke Obat Diabetes,
Integrasi Status PAINS/Brenk, & Pemeringkatan Kandidat Terkalibrasi Medisinal.

Konsep cheminformatics & pemeringkatan transparan (Peer-Review Grade):
1. Morgan Fingerprint (ECFP4-like, radius=2, 2048 bit) untuk 27 senyawa jamu & 9 obat antidiabetes resmi.
2. Kemiripan **Tanimoto** masing-masing senyawa jamu terhadap seluruh obat antidiabetes.
3. Proyeksi ruang kimia 2D menggunakan PCA (Principal Component Analysis).
4. Dua Metrik Pemeringkatan Transparan:
   - `combined_score`: Skor komposit linier klasik (0.45*norm_network + 0.30*norm_qed + 0.25*norm_tanimoto).
   - `medchem_lead_score` (Skor Prioritas Lead Kimia Medisinal):
     * Menggunakan transformasi logaritmik `log1p(raw_network_score)` untuk meredam bias literatur ChEMBL
       pada polifenol yang memiliki puluhan pengujian non-spesifik.
     * Mengkalibrasi Tanimoto terhadap ambang kemiripan ECFP4 absolut (`min(1.0, max_tanimoto / 0.40)`),
       sehingga molekul dengan 0 target eksperimental tidak melompat semu akibat min-max scaling.
     * Memberikan penalti eksplisit terhadap peringatan reaktivitas **PAINS** (-0.18) dan **Brenk** (-0.04/alert),
       serta memisahkan kategori kandidat:
       - `Prioritas Lead (Target Terbukti & Bebas PAINS)`
       - `Polifenol Multi-Target (Perlu Waspada PAINS)`
       - `Kandidat Eksplorasi Baru (0 Target ChEMBL / Research Gap)`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import FilterCatalog, rdFingerprintGenerator
from sklearn.decomposition import PCA

from common import FIG_DIR, PLANT_COLORS, TABLE_DIR

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return MORGAN_GEN.GetFingerprint(mol)


def norm_01(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def evaluasi_pains_brenk_ringkas(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Hitung jumlah peringatan PAINS dan Brenk untuk diintegrasikan langsung ke tabel peringkat."""
    p_pains = FilterCatalog.FilterCatalogParams()
    p_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    cat_pains = FilterCatalog.FilterCatalog(p_pains)

    p_brenk = FilterCatalog.FilterCatalogParams()
    p_brenk.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    cat_brenk = FilterCatalog.FilterCatalog(p_brenk)

    rows = []
    for _, r in comp_df.iterrows():
        mol = Chem.MolFromSmiles(str(r["smiles"]))
        if mol is None:
            continue
        pains_hits = [m.GetDescription() for m in cat_pains.GetMatches(mol)]
        brenk_hits = [m.GetDescription() for m in cat_brenk.GetMatches(mol)]
        if len(pains_hits) == 0 and len(brenk_hits) == 0:
            status = "Bersih (PAINS=0, Brenk=0)"
        elif len(pains_hits) == 0:
            status = f"PAINS Bersih (Brenk={len(brenk_hits)})"
        else:
            status = f"PAINS Alert ({len(pains_hits)}) & Brenk ({len(brenk_hits)})"
        rows.append({
            "compound": r["compound"],
            "n_pains_alerts": len(pains_hits),
            "n_brenk_alerts": len(brenk_hits),
            "medchem_status": status,
        })
    return pd.DataFrame(rows)


def main():
    print("== Langkah 7: Ruang Kimia, Kemiripan Tanimoto & Peringkat Kandidat Terkalibrasi ==")
    comp = pd.read_csv(TABLE_DIR / "01_compounds_pubchem.csv")
    drugs = pd.read_csv(TABLE_DIR / "01_antidiabetic_drugs_pubchem.csv")
    dl = pd.read_csv(TABLE_DIR / "02_druglikeness.csv")
    ct = pd.read_csv(TABLE_DIR / "03_compound_targets.csv")
    hubs = pd.read_csv(TABLE_DIR / "05_hub_genes.csv")

    # 1. Hitung Morgan fingerprints
    jamu_fps = {r["compound"]: smiles_to_fp(r["smiles"]) for _, r in comp.iterrows()}
    drug_fps = {r["drug"]: smiles_to_fp(r["smiles"]) for _, r in drugs.iterrows()}

    # 2. Kemiripan Tanimoto maksimum ke obat antidiabetes
    sim_rows = []
    for cname, fp in jamu_fps.items():
        if fp is None:
            continue
        best_drug, best_sim = None, -1.0
        for dname, dfp in drug_fps.items():
            if dfp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fp, dfp)
            if sim > best_sim:
                best_sim = sim
                best_drug = dname
        sim_rows.append({
            "compound": cname,
            "closest_drug": best_drug,
            "max_tanimoto": round(float(best_sim), 4),
        })
    sim_df = pd.DataFrame(sim_rows)

    # 3. Skor Jaringan Target T2D per senyawa (jumlah target T2D * bobot hub)
    hub_map = hubs.set_index("gene_symbol")["degree"].to_dict()
    ct_t2d = ct[ct["gene_symbol"].isin(hub_map)].copy()
    ct_t2d["target_hub_weight"] = 1.0 + ct_t2d["gene_symbol"].map(hub_map)

    net_score = (
        ct_t2d.groupby("compound")
        .agg(
            n_t2d_targets=("gene_symbol", "nunique"),
            top_t2d_targets=("gene_symbol", lambda s: "; ".join(sorted(set(s))[:6])),
            raw_network_score=("target_hub_weight", "sum"),
        )
        .reset_index()
    )

    alerts_df = evaluasi_pains_brenk_ringkas(comp)

    # 4. Gabungkan semua & hitung Skor Gabungan + Skor Lead Terkalibrasi Medisinal
    rank_df = (
        dl[["compound", "plants", "mw", "clogp", "qed", "lipinski_pass"]]
        .merge(sim_df, on="compound", how="left")
        .merge(net_score, on="compound", how="left")
        .merge(alerts_df, on="compound", how="left")
    )
    rank_df["n_t2d_targets"] = rank_df["n_t2d_targets"].fillna(0).astype(int)
    rank_df["top_t2d_targets"] = rank_df["top_t2d_targets"].fillna("-")
    rank_df["raw_network_score"] = rank_df["raw_network_score"].fillna(0.0)
    rank_df["n_pains_alerts"] = rank_df["n_pains_alerts"].fillna(0).astype(int)
    rank_df["n_brenk_alerts"] = rank_df["n_brenk_alerts"].fillna(0).astype(int)

    rank_df["norm_network"] = norm_01(rank_df["raw_network_score"])
    rank_df["norm_qed"] = norm_01(rank_df["qed"])
    rank_df["norm_tanimoto"] = norm_01(rank_df["max_tanimoto"])
    rank_df["combined_score"] = (
        0.45 * rank_df["norm_network"]
        + 0.30 * rank_df["norm_qed"]
        + 0.25 * rank_df["norm_tanimoto"]
    ).round(4)

    # Skor Lead Medisinal Terkalibrasi (meredam bias jumlah assay & mengintegrasikan penalti PAINS/Brenk)
    log_net = np.log1p(rank_df["raw_network_score"])
    norm_log_net = log_net / max(1e-8, float(log_net.max()))
    cal_tanimoto = np.clip(rank_df["max_tanimoto"] / 0.40, 0.0, 1.0)
    pains_pen = 0.18 * (rank_df["n_pains_alerts"] > 0).astype(float)
    brenk_pen = 0.04 * np.minimum(2, rank_df["n_brenk_alerts"])
    rank_df["medchem_lead_score"] = np.clip(
        0.50 * norm_log_net + 0.35 * rank_df["qed"] + 0.15 * cal_tanimoto - pains_pen - brenk_pen,
        0.0,
        1.0,
    ).round(4)

    def klasifikasi_kandidat(row):
        if row["n_t2d_targets"] > 0 and row["n_pains_alerts"] == 0:
            return "1. Prioritas Lead (Target T2D Terbukti & Bebas PAINS)"
        if row["n_t2d_targets"] > 0 and row["n_pains_alerts"] > 0:
            return "2. Multi-Target Aktif (Waspada Peringatan PAINS)"
        return "3. Kandidat Eksplorasi Baru (0 Target ChEMBL / Research Gap)"

    rank_df["candidate_category"] = rank_df.apply(klasifikasi_kandidat, axis=1)
    rank_df = rank_df.sort_values("combined_score", ascending=False)
    rank_df.to_csv(TABLE_DIR / "07_candidate_ranking.csv", index=False)

    print("  Top 10 Kandidat Senyawa Jamu (Combined Score & MedChem Lead Score):")
    print(
        rank_df[[
            "compound", "plants", "n_t2d_targets", "qed", "max_tanimoto",
            "combined_score", "medchem_lead_score", "medchem_status",
        ]]
        .head(10)
        .to_string(index=False)
    )

    # 5. Plot PCA Ruang Kimia (Jamu + Obat Antidiabetes)
    all_names, all_vecs, all_types, all_labels = [], [], [], []
    for _, r in comp.iterrows():
        fp = jamu_fps.get(r["compound"])
        if fp is not None:
            all_names.append(r["compound"])
            all_vecs.append(np.array(fp))
            all_types.append(r["plants"].split(";")[0].strip())
            all_labels.append("Jamu")
    for _, r in drugs.iterrows():
        fp = drug_fps.get(r["drug"])
        if fp is not None:
            all_names.append(r["drug"])
            all_vecs.append(np.array(fp))
            all_types.append("Obat Antidiabetes")
            all_labels.append("Obat")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(np.vstack(all_vecs))
    var = pca.explained_variance_ratio_ * 100
    pca_df = pd.DataFrame({
        "name": all_names, "PC1": coords[:, 0], "PC2": coords[:, 1],
        "group": all_types, "role": all_labels,
    })
    pca_df.to_csv(TABLE_DIR / "07_chemical_space_pca.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    palette = {**PLANT_COLORS, "Obat Antidiabetes": "#000000"}
    for grp, sub in pca_df.groupby("group"):
        is_drug = grp == "Obat Antidiabetes"
        ax.scatter(
            sub["PC1"], sub["PC2"],
            c=palette.get(grp, "#777777"),
            marker="*" if is_drug else "o",
            s=190 if is_drug else 75,
            label=grp,
            alpha=0.92,
            edgecolors="white" if not is_drug else "gold",
        )
    top_jamu = set(rank_df.head(8)["compound"]).union(
        set(rank_df.sort_values("medchem_lead_score", ascending=False).head(5)["compound"])
    )
    for _, r in pca_df.iterrows():
        if r["role"] == "Obat" or r["name"] in top_jamu:
            ax.annotate(
                r["name"], (r["PC1"], r["PC2"]), xytext=(4, 4),
                textcoords="offset points",
                fontsize=8, fontweight="bold" if r["role"] == "Obat" else "normal",
            )
    ax.set_title("Peta Ruang Kimia (PCA Morgan Fingerprint): Senyawa Jamu vs Obat Antidiabetes", fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% varians)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% varians)")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_chemical_space_pca.png", dpi=220)
    plt.close()
    print("Selesai Langkah 7.")


if __name__ == "__main__":
    main()
