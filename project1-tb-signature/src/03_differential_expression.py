"""
03_differential_expression.py
=============================
Langkah 3: Analisis Ekspresi Diferensial (Differential Gene Expression / DGE).

Pertanyaan biologis:
"Gen mana saja di darah yang aktivitasnya berbeda secara signifikan antara
pasien TBC Aktif (Active_TB) dibandingkan kelompok kontrol (Latent_TB + Healthy)?"

Metode statistik:
1. Hitung log2 Fold Change (log2FC) = rata-rata ekspresi Active_TB - rata-rata Non-TB
   (karena data sudah berskala log2, pengurangan = log2 dari rasio!).
2. Uji Welch's t-test untuk setiap gen (~9.000 gen).
3. Koreksi pengujian berganda (multiple testing correction) dengan metode
   Benjamini-Hochberg untuk mendapatkan FDR (False Discovery Rate / q-value).
   Kenapa wajib FDR? Karena kalau menguji 9.000 gen dengan p < 0.05 tanpa koreksi,
   secara kebetulan saja sudah ada ~450 hasil positif palsu!
4. Visualisasi:
   - Volcano Plot (log2FC vs -log10 FDR) + label gen top (mis. GBP1, BATF2, FCGR1A, dll.)
   - Clustered Heatmap 30 gen paling signifikan.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"


def main():
    print("== Langkah 3: Analisis Ekspresi Diferensial (DGE) ==")
    expr = pd.read_parquet(PROC_DIR / "discovery_expr.parquet")
    meta = pd.read_csv(PROC_DIR / "discovery_meta.csv", index_col=0)

    tb_idx = meta.index[meta["y_tb"] == 1]
    ctrl_idx = meta.index[meta["y_tb"] == 0]
    print(f"  Sampel Discovery: {len(tb_idx)} Active TB vs {len(ctrl_idx)} Non-TB (Latent/Healthy)")

    tb_vals = expr[tb_idx].values
    ctrl_vals = expr[ctrl_idx].values

    mean_tb = tb_vals.mean(axis=1)
    mean_ctrl = ctrl_vals.mean(axis=1)
    log2fc = mean_tb - mean_ctrl

    # Welch's t-test (tidak mengasumsikan varians kedua kelompok sama)
    t_stat, p_vals = stats.ttest_ind(tb_vals, ctrl_vals, axis=1, equal_var=False)
    # Koreksi Benjamini-Hochberg (FDR)
    _, fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    de = pd.DataFrame({
        "gene": expr.index,
        "mean_active_tb": mean_tb,
        "mean_control": mean_ctrl,
        "log2FC": log2fc,
        "abs_log2FC": np.abs(log2fc),
        "t_stat": t_stat,
        "p_value": p_vals,
        "fdr": fdr,
    })

    # Kriteria gen signifikan: FDR < 0.05 dan |log2FC| >= 0.5 (~1.41x perubahan)
    de["status"] = "Tidak Signifikan"
    de.loc[(de["fdr"] < 0.05) & (de["log2FC"] >= 0.5), "status"] = "Naik di TB (Up)"
    de.loc[(de["fdr"] < 0.05) & (de["log2FC"] <= -0.5), "status"] = "Turun di TB (Down)"

    de = de.sort_values(["fdr", "abs_log2FC"], ascending=[True, False])
    de.to_csv(TABLE_DIR / "03_differential_expression.csv", index=False)

    n_up = (de["status"] == "Naik di TB (Up)").sum()
    n_down = (de["status"] == "Turun di TB (Down)").sum()
    print(f"  Gen signifikan (FDR < 0.05, |log2FC| >= 0.5): {n_up + n_down} total ({n_up} Up, {n_down} Down)")

    top30 = de[de["status"] != "Tidak Signifikan"].head(30)
    top30.to_csv(TABLE_DIR / "03_top30_degs.csv", index=False)
    print("  Top 10 Gen:")
    print(top30[["gene", "log2FC", "fdr", "status"]].head(10).to_string(index=False))

    # --- 1. Volcano Plot ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    de["neg_log10_fdr"] = -np.log10(np.clip(de["fdr"], 1e-300, 1.0))

    colors = {
        "Naik di TB (Up)": "#d62728",
        "Turun di TB (Down)": "#1f77b4",
        "Tidak Signifikan": "#c7c7c7",
    }
    for st, col in colors.items():
        sub = de[de["status"] == st]
        ax.scatter(
            sub["log2FC"], sub["neg_log10_fdr"],
            c=col, label=f"{st} ({len(sub)})", s=16 if st == "Tidak Signifikan" else 28,
            alpha=0.45 if st == "Tidak Signifikan" else 0.85, edgecolors="none"
        )

    ax.axhline(-np.log10(0.05), ls="--", lw=1, color="#444444")
    ax.axvline(0.5, ls="--", lw=1, color="#444444")
    ax.axvline(-0.5, ls="--", lw=1, color="#444444")

    # Beri label pada 12 gen paling ekstrem
    label_genes = pd.concat([
        de[de["status"] == "Naik di TB (Up)"].head(8),
        de[de["status"] == "Turun di TB (Down)"].head(4),
    ])
    for _, r in label_genes.iterrows():
        ax.annotate(
            r["gene"],
            (r["log2FC"], r["neg_log10_fdr"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#222222",
        )

    ax.set_title("Volcano Plot: Active TB vs Kontrol (Discovery Cohort)", fontsize=13, fontweight="bold")
    ax.set_xlabel("log2 Fold Change (Active TB / Kontrol)")
    ax.set_ylabel("-log10(FDR)")
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_volcano_plot.png", dpi=220)
    plt.close()

    # --- 2. Heatmap Top 30 Gen ---
    top_genes = top30["gene"].tolist()
    mat = expr.loc[top_genes]
    # Z-score per baris (gen) supaya pola naik/turun terlihat kontras
    mat_z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1) + 1e-8, axis=0)
    # Urutkan sampel berdasarkan kelompok agar pola mudah dibaca
    order = meta.sort_values(["y_tb", "group"], ascending=[False, True]).index
    mat_z = mat_z[order]

    group_palette = {"Active_TB": "#d62728", "Latent_TB": "#ff7f0e", "Healthy": "#2ca02c"}
    col_colors = meta.loc[order, "group"].map(group_palette)

    g = sns.clustermap(
        mat_z,
        col_cluster=False,
        row_cluster=True,
        col_colors=col_colors,
        cmap="RdBu_r",
        vmin=-2.5, vmax=2.5,
        xticklabels=False,
        yticklabels=True,
        figsize=(11, 8),
        cbar_kws={"label": "Z-score Ekspresi"},
    )
    g.fig.suptitle("Heatmap Top 30 Gen Terdiferensiasi (Active TB vs Kontrol)", y=1.02, fontweight="bold")
    # Tambahkan legenda manual untuk bar warna kolom
    for label, col in group_palette.items():
        g.ax_col_dendrogram.bar(0, 0, color=col, label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=3, frameon=False)
    g.savefig(FIG_DIR / "03_heatmap_top30.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Selesai! Grafik tersimpan di:", FIG_DIR)


if __name__ == "__main__":
    main()
