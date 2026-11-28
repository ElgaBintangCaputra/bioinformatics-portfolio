"""
03_distance_barcode_gap.py
==========================
Langkah 3: Menghitung Jarak Genetik Kimura 2-Parameter (K2P) & Analisis Barcode Gap.

Konsep inti DNA Barcoding:
- Jarak K2P membedakan mutasi *transisi* (A<->G atau C<->T, lebih sering terjadi)
  dan *transversi* (purin <-> pirimidin, lebih jarang). Ini standar resmi di BOLD Systems!
- Intraspecific distance: jarak genetik antar-individu dalam SPESIES YANG SAMA (seharusnya kecil, < 1-2%).
- Interspecific distance: jarak genetik antar-spesies BERBEDA (seharusnya lebih besar).
- Barcode Gap per spesies = (Jarak terdekat ke spesies lain) - (Jarak terjauh di dalam spesies sendiri).
  Jika Barcode Gap > 0, spesies tersebut dapat dibedakan dengan tegas oleh DNA barcode COI!
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import AlignIO

from common import get_paths, load_config, setup_logger

PURINES = {"A", "G"}
PYRIMIDINES = {"C", "T"}
VALID_BASES = {"A", "C", "G", "T"}


def k2p_distance(seq1: str, seq2: str) -> float:
    """Menghitung jarak evolusi Kimura 2-Parameter (K2P) dengan pairwise deletion untuk gap/N."""
    n_comp = 0
    transitions = 0
    transversions = 0

    for a, b in zip(seq1, seq2):
        if a not in VALID_BASES or b not in VALID_BASES:
            continue
        n_comp += 1
        if a != b:
            if (a in PURINES and b in PURINES) or (a in PYRIMIDINES and b in PYRIMIDINES):
                transitions += 1
            else:
                transversions += 1

    if n_comp == 0:
        return np.nan
    p = transitions / n_comp
    q = transversions / n_comp
    t1 = 1.0 - 2.0 * p - q
    t2 = 1.0 - 2.0 * q
    if t1 <= 0 or t2 <= 0:
        # Fallback ke p-distance jika jenuh (sangat jarang untuk COI famili ikan)
        return (transitions + transversions) / n_comp
    return max(0.0, -0.5 * math.log(t1) - 0.25 * math.log(t2))


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    logger.info("== Langkah 3: Jarak Genetik K2P & Barcode Gap ==")

    aln = AlignIO.read(paths["processed_dir"] / "coi_aligned_trimmed.fasta", "fasta")
    ids = [r.id for r in aln]
    seqs = [str(r.seq).upper() for r in aln]
    species_list = [rid.split("|")[1].replace("_", " ") for rid in ids]
    family_list = [rid.split("|")[2] for rid in ids]
    n = len(ids)

    dist_mat = np.zeros((n, n))
    intra_vals = []
    inter_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            d = k2p_distance(seqs[i], seqs[j])
            dist_mat[i, j] = dist_mat[j, i] = d
            if species_list[i] == species_list[j]:
                intra_vals.append(d)
            else:
                inter_vals.append(d)

    # Simpan matriks jarak K2P penuh untuk langkah filogenetik
    dist_df = pd.DataFrame(dist_mat, index=ids, columns=ids)
    dist_df.to_csv(paths["tables_dir"] / "03_k2p_distance_matrix.csv")

    # Hitung Barcode Gap per spesies
    gap_rows = []
    unique_species = sorted(set(species_list))
    sp_arr = np.array(species_list)
    fam_map = dict(zip(species_list, family_list))

    for sp in unique_species:
        idx_sp = np.where(sp_arr == sp)[0]
        idx_other = np.where(sp_arr != sp)[0]

        sub_intra = dist_mat[np.ix_(idx_sp, idx_sp)]
        # Ambil segitiga atas (pasangan berbeda dalam spesies yang sama)
        iu = np.triu_indices(len(idx_sp), k=1)
        intra_dists = sub_intra[iu]
        max_intra = float(np.max(intra_dists)) if len(intra_dists) > 0 else 0.0
        mean_intra = float(np.mean(intra_dists)) if len(intra_dists) > 0 else 0.0

        sub_inter = dist_mat[np.ix_(idx_sp, idx_other)]
        min_inter = float(np.min(sub_inter))
        # Cari spesies tetangga terdekat (Nearest Neighbor)
        min_pos = np.unravel_index(np.argmin(sub_inter), sub_inter.shape)
        nn_species = species_list[idx_other[min_pos[1]]]

        gap = min_inter - max_intra
        gap_rows.append({
            "species": sp,
            "family": fam_map[sp],
            "n_seq": len(idx_sp),
            "mean_intra_k2p": round(mean_intra, 4),
            "max_intra_k2p": round(max_intra, 4),
            "min_inter_k2p": round(min_inter, 4),
            "nearest_neighbor": nn_species,
            "barcode_gap": round(gap, 4),
            "has_gap": gap > 0,
        })

    gap_df = pd.DataFrame(gap_rows).sort_values("barcode_gap", ascending=False)
    gap_df.to_csv(paths["tables_dir"] / "03_barcode_gap_per_species.csv", index=False)

    summary = pd.DataFrame([
        {"metrik": "Jumlah Sekuens", "nilai": n},
        {"metrik": "Jumlah Spesies", "nilai": len(unique_species)},
        {"metrik": "Rata-rata Jarak Intraspesifik (K2P)", "nilai": round(float(np.mean(intra_vals)), 4)},
        {"metrik": "Maksimum Jarak Intraspesifik (K2P)", "nilai": round(float(np.max(intra_vals)), 4)},
        {"metrik": "Rata-rata Jarak Interspesifik (K2P)", "nilai": round(float(np.mean(inter_vals)), 4)},
        {"metrik": "Minimum Jarak Interspesifik (K2P)", "nilai": round(float(np.min(inter_vals)), 4)},
        {"metrik": "Spesies dengan Barcode Gap > 0", "nilai": f"{gap_df['has_gap'].sum()}/{len(gap_df)}"},
    ])
    summary.to_csv(paths["tables_dir"] / "03_distance_summary.csv", index=False)
    logger.info("\n" + summary.to_string(index=False))

    # --- Plot Barcode Gap (2 panel) ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel A: Histogram Intraspesifik vs Interspesifik
    axes[0].hist(intra_vals, bins=30, density=True, alpha=0.75, color="#2ca02c", label=f"Intraspesifik (n={len(intra_vals)})")
    axes[0].hist(inter_vals, bins=40, density=True, alpha=0.65, color="#1f77b4", label=f"Interspesifik (n={len(inter_vals)})")
    axes[0].axvline(0.02, color="#d62728", ls="--", lw=1.2, label="Ambang klasik 2% (0.02)")
    axes[0].set_title("Distribusi Jarak Genetik K2P", fontweight="bold")
    axes[0].set_xlabel("Jarak Genetik K2P")
    axes[0].set_ylabel("Kepadatan (Density)")
    axes[0].legend(fontsize=8)

    # Panel B: Max Intra vs Min Inter (Scatter: di atas garis y=x berarti punya Barcode Gap)
    fam_colors = {"Lutjanidae": "#d62728", "Epinephelidae": "#2ca02c", "Scombridae": "#1f77b4"}
    for fam, col in fam_colors.items():
        sub = gap_df[gap_df["family"] == fam]
        axes[1].scatter(sub["max_intra_k2p"], sub["min_inter_k2p"], c=col, label=fam, s=70, alpha=0.88, edgecolors="white")
    lim = max(gap_df["max_intra_k2p"].max(), gap_df["min_inter_k2p"].max()) * 1.1
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="y = x (Gap = 0)")
    # Beri label 5 spesies dengan gap terkecil / berpotensi tumpang tindih
    for _, r in gap_df.tail(4).iterrows():
        axes[1].annotate(r["species"], (r["max_intra_k2p"], r["min_inter_k2p"]), xytext=(5, -3),
                         textcoords="offset points", fontsize=7.5)
    axes[1].set_title("Barcode Gap per Spesies (Titik di Atas Garis = Terpisah Tegas)", fontweight="bold")
    axes[1].set_xlabel("Jarak Intraspesifik Maksimum (K2P)")
    axes[1].set_ylabel("Jarak ke Tetangga Terdekat / Min Interspesifik (K2P)")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "03_barcode_gap_plot.png", dpi=220)
    plt.close()
    logger.info("Selesai Langkah 3.")


if __name__ == "__main__":
    main()
