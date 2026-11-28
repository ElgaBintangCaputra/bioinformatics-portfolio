"""
08_qsar_3d_conformers.py
========================
Langkah 8 (Pendalaman Cheminformatics, QSAR ML + Applicability Domain, & Ensemble Konformer 3D):
1. Structural Alerts (PAINS & Brenk) dengan RDKit FilterCatalog:
   Mendeteksi gugus kimia reaktif / berpotensi positif palsu pada uji lab (PAINS) maupun
   gugus toksikofor/reaktif medisinal (Brenk) secara terpisah dan transparan.
2. Model QSAR Machine Learning Nyata untuk Target DPP4 Manusia (CHEMBL284):
   - Mengunduh senyawa inhibitor DPP4 berlabel pChEMBL dari ChEMBL (di-cache lokal, 392 senyawa unik).
   - Evaluasi ketat dengan **Bemis-Murcko Scaffold Split**, **5-Fold GroupKFold by Scaffold**,
     serta **5-Fold Stratified CV**.
   - **Kuantifikasi Applicability Domain (AD):** Menghitung kemiripan Tanimoto terdekat
     masing-masing senyawa jamu terhadap ruang kimia inhibitor DPP4 ChEMBL (`max_tanimoto_dpp4_train`)
     untuk membedakan prediksi interpolasi vs ekstrapolasi produk alam.
3. Generasi Ensemble Konformer 3D Deterministik (10 Konformer ETKDGv3 + Minimisasi MMFF94):
   Memilih konformer berenergi potensial terendah dan menyimpan struktur 3D siap-docking (.sdf).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, FilterCatalog, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from common import FIG_DIR, ROOT, TABLE_DIR, cached_request

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
STRUCT_DIR = ROOT / "results" / "structures"
STRUCT_DIR.mkdir(parents=True, exist_ok=True)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def cek_pains_brenk(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluasi peringatan struktur medisinal (PAINS & Brenk) untuk semua senyawa jamu."""
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
        rows.append({
            "compound": r["compound"],
            "n_pains_alerts": len(pains_hits),
            "pains_detail": "; ".join(pains_hits) if pains_hits else "Bersih",
            "n_brenk_alerts": len(brenk_hits),
            "brenk_detail": "; ".join(brenk_hits[:3]) if brenk_hits else "Bersih",
        })
    return pd.DataFrame(rows)


def unduh_dataset_qsar_dpp4(max_records: int = 500) -> pd.DataFrame:
    """Ambil data bioaktivitas DPP4 manusia (CHEMBL284) dari ChEMBL API (dengan cache lokal)."""
    rows = []
    for offset in range(0, max_records, 100):
        resp = cached_request(
            "chembl_qsar",
            f"{CHEMBL_API}/activity.json",
            params={
                "target_chembl_id": "CHEMBL284",
                "pchembl_value__isnull": "false",
                "limit": 100,
                "offset": offset,
                "format": "json",
            },
        )
        batch = (resp or {}).get("activities", [])
        if not batch:
            break
        for a in batch:
            smi = a.get("canonical_smiles")
            pval = a.get("pchembl_value")
            if smi and pval is not None:
                rows.append({"smiles": smi, "pchembl": float(pval)})

    df = pd.DataFrame(rows)
    canon_rows = []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is not None:
            csmi = Chem.MolToSmiles(m, isomericSmiles=False)
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
            canon_rows.append({"canon_smiles": csmi, "scaffold": scaf, "pchembl": r["pchembl"]})

    clean = (
        pd.DataFrame(canon_rows)
        .groupby(["canon_smiles", "scaffold"], as_index=False)["pchembl"]
        .median()
    )
    clean["active"] = (clean["pchembl"] >= 6.0).astype(int)
    return clean


def scaffold_split(df: pd.DataFrame, test_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Memisahkan indeks latih dan uji berdasarkan kelompok Murcko Scaffold (anti-kebocoran struktur)."""
    groups = df.groupby("scaffold").indices
    sorted_scafs = sorted(groups.values(), key=len)
    n_test_target = int(len(df) * test_frac)
    test_idx, train_idx = [], []
    for idxs in sorted_scafs:
        if len(test_idx) < n_test_target:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return np.array(train_idx), np.array(test_idx)


def bangkitkan_konformer_3d(comp_df: pd.DataFrame, top_names: list[str], n_confs: int = 10) -> pd.DataFrame:
    """
    Bangkitkan ensemble n_confs struktur 3D secara deterministik (ETKDGv3, randomSeed=42),
    minimisasi seluruh konformer dengan medan gaya MMFF94, dan simpan konformer berenergi terendah ke .sdf.
    """
    rows = []
    for name in top_names:
        sub = comp_df[comp_df["compound"] == name]
        if sub.empty:
            continue
        mol = Chem.AddHs(Chem.MolFromSmiles(str(sub["smiles"].iloc[0])))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
        if not conf_ids:
            cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            conf_ids = [cid] if cid >= 0 else []

        energies = []
        best_cid = conf_ids[0] if conf_ids else -1
        min_e = np.nan
        if conf_ids and AllChem.MMFFHasAllMoleculeParams(mol):
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            for cid in conf_ids:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                if ff is not None:
                    ff.Minimize(maxIts=500)
                    e = float(ff.CalcEnergy())
                    energies.append((e, cid))
            if energies:
                energies.sort(key=lambda x: x[0])
                min_e, best_cid = energies[0]

        span_e = (energies[-1][0] - energies[0][0]) if len(energies) > 1 else 0.0
        safe_fn = name.replace(" ", "_").replace("/", "_") + ".sdf"
        writer = Chem.SDWriter(str(STRUCT_DIR / safe_fn))
        mol.SetProp("_Name", name)
        if not np.isnan(min_e):
            mol.SetProp("MMFF94_Min_Energy_kcal_mol", f"{min_e:.2f}")
        if best_cid >= 0:
            writer.write(mol, confId=best_cid)
        else:
            writer.write(mol)
        writer.close()

        rows.append({
            "compound": name,
            "n_atoms_with_H": mol.GetNumAtoms(),
            "n_conformers_sampled": len(conf_ids),
            "mmff94_energy_kcal_mol": round(float(min_e), 2) if not np.isnan(min_e) else np.nan,
            "conformer_energy_span_kcal_mol": round(float(span_e), 2),
            "sdf_file": f"results/structures/{safe_fn}",
        })
    return pd.DataFrame(rows)


def main():
    print("== Langkah 8: PAINS/Brenk Alerts, QSAR ML DPP4 (+ Applicability Domain), & Konformer 3D ==")
    comp_df = pd.read_csv(TABLE_DIR / "01_compounds_pubchem.csv")
    rank_df = pd.read_csv(TABLE_DIR / "07_candidate_ranking.csv")

    # 1. Structural Alerts (PAINS & Brenk)
    alerts_df = cek_pains_brenk(comp_df)
    alerts_df.to_csv(TABLE_DIR / "08_pains_brenk_alerts.csv", index=False)
    n_pains = int((alerts_df["n_pains_alerts"] > 0).sum())
    n_brenk = int((alerts_df["n_brenk_alerts"] > 0).sum())
    print(f"  PAINS alerts: {n_pains}/{len(alerts_df)} senyawa | Brenk alerts: {n_brenk}/{len(alerts_df)} senyawa.")

    # 2. QSAR ML untuk target DPP4 (CHEMBL284)
    print("  Mengambil dataset QSAR DPP4 manusia (CHEMBL284)...")
    qsar_df = unduh_dataset_qsar_dpp4(max_records=500)
    train_fps = [MORGAN_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in qsar_df["canon_smiles"]]
    X_q = np.vstack([np.array(fp) for fp in train_fps])
    y_reg = qsar_df["pchembl"].values
    y_cls = qsar_df["active"].values
    scaf_groups = qsar_df["scaffold"].values

    tr_idx, te_idx = scaffold_split(qsar_df, test_frac=0.25)
    reg = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    reg.fit(X_q[tr_idx], y_reg[tr_idx])
    pred_reg = reg.predict(X_q[te_idx])
    rmse = float(np.sqrt(mean_squared_error(y_reg[te_idx], pred_reg)))
    r2 = float(r2_score(y_reg[te_idx], pred_reg))
    pear_r = float(pearsonr(y_reg[te_idx], pred_reg)[0])

    cls_model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    cls_model.fit(X_q[tr_idx], y_cls[tr_idx])
    scaf_auc = float(roc_auc_score(y_cls[te_idx], cls_model.predict_proba(X_q[te_idx])[:, 1]))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_q, y_cls, cv=cv, method="predict_proba",
    )[:, 1]
    cv_auc = float(roc_auc_score(y_cls, cv_probs))

    # Evaluasi tambahan: 5-Fold GroupKFold berdasarkan Murcko Scaffold
    gkf = GroupKFold(n_splits=5)
    gkf_probs = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_q, y_cls, cv=gkf, groups=scaf_groups, method="predict_proba",
    )[:, 1]
    gkf_auc = float(roc_auc_score(y_cls, gkf_probs))

    metrics_df = pd.DataFrame([{
        "Target": "DPP4 (CHEMBL284)",
        "N_Senyawa_Unik": len(qsar_df),
        "N_Scaffold_Unik": qsar_df["scaffold"].nunique(),
        "CV5_ROC_AUC": round(cv_auc, 4),
        "GroupCV5_Scaffold_ROC_AUC": round(gkf_auc, 4),
        "ScaffoldSplit_ROC_AUC": round(scaf_auc, 4),
        "ScaffoldSplit_R2": round(r2, 4),
        "ScaffoldSplit_RMSE": round(rmse, 4),
        "ScaffoldSplit_Pearson_r": round(pear_r, 4),
    }])
    metrics_df.to_csv(TABLE_DIR / "08_qsar_metrics.csv", index=False)
    print("  Performa QSAR DPP4:")
    print(metrics_df.to_string(index=False))

    # Latih final di seluruh data QSAR DPP4 & prediksi 27 senyawa jamu + Applicability Domain (AD)
    reg_full = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1).fit(X_q, y_reg)
    cls_full = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1).fit(X_q, y_cls)

    jamu_mols = [Chem.MolFromSmiles(s) for s in comp_df["smiles"]]
    jamu_fps = [MORGAN_GEN.GetFingerprint(m) for m in jamu_mols]
    X_jamu = np.vstack([np.array(fp) for fp in jamu_fps])

    # Hitung kemiripan Tanimoto maksimum ke set latih DPP4 untuk menilai Applicability Domain
    max_sim_dpp4 = []
    for jfp in jamu_fps:
        sims = DataStructs.BulkTanimotoSimilarity(jfp, train_fps)
        max_sim_dpp4.append(round(float(max(sims)), 4) if sims else 0.0)

    jamu_pred = pd.DataFrame({
        "compound": comp_df["compound"],
        "plants": comp_df["plants"],
        "pred_dpp4_pchembl": np.round(reg_full.predict(X_jamu), 3),
        "pred_dpp4_prob_active": np.round(cls_full.predict_proba(X_jamu)[:, 1], 3),
        "max_tanimoto_dpp4_train": max_sim_dpp4,
        "applicability_domain": [
            "Inside AD (>=0.25)" if s >= 0.25 else "Extrapolation / Natural Product (<0.25)"
            for s in max_sim_dpp4
        ],
    }).merge(
        alerts_df[["compound", "n_pains_alerts", "pains_detail", "n_brenk_alerts", "brenk_detail"]],
        on="compound",
    )
    jamu_pred = jamu_pred.sort_values("pred_dpp4_pchembl", ascending=False)
    jamu_pred.to_csv(TABLE_DIR / "08_jamu_qsar_predictions.csv", index=False)
    print("\n  Top 8 Prediksi QSAR DPP4 Senyawa Jamu (+ Applicability Domain):")
    print(jamu_pred.head(8).to_string(index=False))

    # 3. Konformer 3D untuk Top 5 kandidat ranking akhir
    top5 = rank_df["compound"].head(5).tolist()
    conf_df = bangkitkan_konformer_3d(comp_df, top5, n_confs=10)
    conf_df.to_csv(TABLE_DIR / "08_3d_conformers.csv", index=False)
    print("\n  Ensemble Konformer 3D (10 Konformer ETKDGv3 + MMFF94 Terendah) tersimpan:")
    print(conf_df.to_string(index=False))

    # Plot 2 panel: (Kiri) Scaffold-Split Predicted vs Actual pChEMBL DPP4, (Kanan) Prediksi QSAR Jamu + PAINS
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    axes[0].scatter(y_reg[te_idx], pred_reg, alpha=0.65, color="#1f77b4", edgecolors="white", s=45)
    lims = [min(y_reg[te_idx].min(), pred_reg.min()) - 0.3, max(y_reg[te_idx].max(), pred_reg.max()) + 0.3]
    axes[0].plot(lims, lims, "r--", lw=1.3, label="Ideal (y = x)")
    axes[0].set_title(
        f"QSAR DPP4 (Scaffold Split): r={pear_r:.2f}, RMSE={rmse:.2f}, AUC={scaf_auc:.2f}",
        fontweight="bold",
    )
    axes[0].set_xlabel("pChEMBL Eksperimental (ChEMBL)")
    axes[0].set_ylabel("pChEMBL Prediksi Random Forest")
    axes[0].legend()

    top12_q = jamu_pred.head(12).iloc[::-1]
    bar_cols = ["#d62728" if p > 0 else "#2ca02c" for p in top12_q["n_pains_alerts"]]
    axes[1].barh(top12_q["compound"], top12_q["pred_dpp4_pchembl"], color=bar_cols, alpha=0.88)
    axes[1].axvline(6.0, ls="--", color="#333333", lw=1, label="Ambang Aktif Sintetis (pChEMBL=6)")
    axes[1].barh([], [], color="#2ca02c", label="Bersih PAINS")
    axes[1].barh([], [], color="#d62728", label="Ada Peringatan PAINS")
    axes[1].set_title("Prediksi QSAR DPP4 pada Senyawa Jamu & Status PAINS", fontweight="bold")
    axes[1].set_xlabel("Prediksi pChEMBL (DPP4)")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_qsar_pains_summary.png", dpi=220)
    plt.close()
    print("Selesai Langkah 8.")


if __name__ == "__main__":
    main()
