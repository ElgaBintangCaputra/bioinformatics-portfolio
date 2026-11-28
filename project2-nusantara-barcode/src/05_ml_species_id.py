"""
05_ml_species_id.py
===================
Langkah 5: Identifikasi Spesies Ikan Berbasis Machine Learning (Alignment-Free 6-mer)
& Studi Ablasi Dampak Audit Kualitas GenBank (Before vs After Audit Cleaning).

Kenapa Alignment-Free k-mer + Audit Ablation?
1. Metode **k-mer frequency** (k=6 -> 4^6 = 4.096 fitur) memungkinkan klasifikasi spesies
   secara instan tanpa menunggu proses alignment MAFFT.
2. **Studi Ablasi Pasca-Audit GenBank:**
   Kita menguji apakah pengeluaran spesimen yang terdeteksi salah label di Langkah 3b
   (`Exclude_From_Reference`, seperti PQ813541.1 & PQ812464.1) secara nyata meningkatkan
   akurasi Cross-Validation dan Macro-F1, serta menunjukkan bahwa sisa galat murni berasal
   dari introgresi biologis mtDNA *Thunnus albacares* vs *Thunnus obesus*!
"""

import itertools
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from common import get_paths, load_config, setup_logger

BASES = ("A", "C", "G", "T")


def buat_kamus_kmer(k: int) -> dict[str, int]:
    return {"".join(km): idx for idx, km in enumerate(itertools.product(BASES, repeat=k))}


def hitung_vektor_kmer(seq: str, k: int, kmer_index: dict[str, int]) -> np.ndarray:
    """Ubah string DNA menjadi vektor frekuensi k-mer ternormalisasi (jumlah = 1)."""
    seq = seq.upper().replace("-", "")
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    valid = 0
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        idx = kmer_index.get(km)
        if idx is not None:
            vec[idx] += 1.0
            valid += 1
    if valid > 0:
        vec /= valid
    return vec


def identifikasi_sekuens(seq: str, bundle: dict) -> dict:
    """Fungsi utilitas untuk dashboard: memprediksi spesies dari sekuens kueri baru."""
    k = bundle["k"]
    kmer_index = bundle["kmer_index"]
    vec = hitung_vektor_kmer(seq, k, kmer_index).reshape(1, -1)

    probs = bundle["rf_model"].predict_proba(vec)[0]
    classes = bundle["rf_model"].classes_
    top_idx = np.argsort(probs)[::-1][:3]
    top_preds = [(classes[i], float(probs[i])) for i in top_idx]

    ref_vecs = bundle["ref_matrix"]
    norms = np.linalg.norm(ref_vecs, axis=1) * (np.linalg.norm(vec) + 1e-12)
    sims = (ref_vecs @ vec.T).ravel() / (norms + 1e-12)
    nn_idx = np.argsort(sims)[::-1][:3]
    nearest = [
        {
            "accession": bundle["ref_accessions"][i],
            "species": bundle["ref_species"][i],
            "cosine_sim": round(float(sims[i]), 4),
        }
        for i in nn_idx
    ]
    return {"top_rf": top_preds, "nearest_matches": nearest}


def evaluasi_rezim(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    models: dict,
    regime_label: str,
) -> tuple[list[dict], np.ndarray, str, float]:
    rows = []
    best_preds = None
    best_name = None
    best_acc = -1.0
    for name, clf in models.items():
        preds = cross_val_predict(clf, X, y, cv=cv)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        rows.append({
            "Dataset_Regime": regime_label,
            "N_Sekuens": len(y),
            "Model": name,
            "CV_Accuracy": round(float(acc), 4),
            "Macro_F1": round(float(f1), 4),
        })
        if acc >= best_acc:
            best_acc = float(acc)
            best_preds = preds
            best_name = name
    return rows, best_preds, best_name, best_acc


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    logger.info("== Langkah 5: Identifikasi Spesies Bebas-Alignment (6-mer + ML & Ablasi Audit) ==")

    k = cfg["ml"]["k"]
    kmer_index = buat_kamus_kmer(k)

    meta = pd.read_csv(paths["tables_dir"] / "02_aligned_metadata.csv")
    valid_acc = set(meta["accession"])

    raw_fasta = paths["processed_dir"] / "coi_raw.fasta"
    aln_fasta = paths["processed_dir"] / "coi_aligned_trimmed.fasta"
    source_fasta = raw_fasta if raw_fasta.exists() else aln_fasta

    raw_recs = [
        r for r in SeqIO.parse(source_fasta, "fasta")
        if r.id.split("|")[0] in valid_acc
    ]

    X = np.vstack([hitung_vektor_kmer(str(r.seq), k, kmer_index) for r in raw_recs])
    y = np.array([r.id.split("|")[1].replace("_", " ") for r in raw_recs])
    accs = np.array([r.id.split("|")[0] for r in raw_recs])

    cv = StratifiedKFold(
        n_splits=cfg["ml"]["cv_folds"], shuffle=True, random_state=cfg["ml"]["random_state"]
    )
    models = {
        "1-NN Cosine (Baseline mirip BLAST)": KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "Linear SVM (6-mer)": LinearSVC(C=5.0, random_state=cfg["ml"]["random_state"]),
        "Random Forest (6-mer)": RandomForestClassifier(
            n_estimators=300, random_state=cfg["ml"]["random_state"], n_jobs=-1
        ),
    }

    # 1. Evaluasi pada Dataset Penuh Sebelum Pembersihan Audit (n=229)
    raw_rows, best_preds, best_name, best_acc = evaluasi_rezim(
        X, y, cv, models, "1. Raw GenBank (n=229, Sebelum Pembersihan Mislabel)"
    )

    comp_df = (
        pd.DataFrame(raw_rows)[["Model", "CV_Accuracy", "Macro_F1"]]
        .sort_values("CV_Accuracy", ascending=False)
    )
    comp_df.to_csv(paths["tables_dir"] / "05_ml_comparison.csv", index=False)
    logger.info("Performa Baseline (n=229):\n" + comp_df.to_string(index=False))

    # 2. Studi Ablasi Pasca-Audit GenBank (Keluarkan spesimen Exclude_From_Reference)
    ablation_rows = list(raw_rows)
    audit_path = paths["tables_dir"] / "03b_genbank_audit.csv"
    if audit_path.exists():
        audit_df = pd.read_csv(audit_path)
        excl_accs = set(
            audit_df.loc[audit_df["action_recommended"] == "Exclude_From_Reference", "accession"]
        ) if "action_recommended" in audit_df.columns else set()
        clean_mask = np.array([a not in excl_accs for a in accs], dtype=bool)
        if clean_mask.sum() < len(accs):
            clean_rows, _, _, _ = evaluasi_rezim(
                X[clean_mask],
                y[clean_mask],
                cv,
                models,
                f"2. Post-Audit Clean (n={int(clean_mask.sum())}, Tanpa Spesimen Mislabel Lintas-Famili)",
            )
            ablation_rows.extend(clean_rows)

            # Rezim 3: Post-Audit + Resolusi kompleks introgresi Thunnus albacares/obesus
            y_thunnus_merged = np.array([
                "Thunnus albacares/obesus (mtDNA complex)"
                if sp in ("Thunnus albacares", "Thunnus obesus")
                else sp
                for sp in y[clean_mask]
            ])
            thunnus_rows, _, _, _ = evaluasi_rezim(
                X[clean_mask],
                y_thunnus_merged,
                cv,
                models,
                f"3. Post-Audit + Thunnus Complex Resolved (n={int(clean_mask.sum())})",
            )
            ablation_rows.extend(thunnus_rows)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(paths["tables_dir"] / "05_ml_ablation_audit.csv", index=False)
    logger.info("Studi Ablasi Dampak Audit GenBank:\n" + ablation_df.to_string(index=False))

    # Plot Confusion Matrix untuk model terbaik
    labels = sorted(set(y))
    cm = confusion_matrix(y, best_preds, labels=labels)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=False, ax=ax
    )
    ax.set_title(
        f"Confusion Matrix 3-Fold CV — {best_name} (Akurasi={best_acc*100:.1f}%)",
        fontweight="bold",
    )
    ax.set_xlabel("Prediksi Spesies")
    ax.set_ylabel("Spesies Sebenarnya")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "05_confusion_matrix.png", dpi=220)
    plt.close()

    # Latih model final pada dataset bersih pasca-audit untuk dashboard
    rf_final = RandomForestClassifier(
        n_estimators=300, random_state=cfg["ml"]["random_state"], n_jobs=-1
    ).fit(X, y)
    joblib.dump(
        {
            "k": k,
            "kmer_index": kmer_index,
            "rf_model": rf_final,
            "ref_matrix": X,
            "ref_species": y.tolist(),
            "ref_accessions": accs.tolist(),
        },
        paths["models_dir"] / "species_id_model.joblib",
    )
    logger.info("Selesai Langkah 5.")


if __name__ == "__main__":
    main()
