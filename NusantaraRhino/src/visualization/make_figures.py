"""
make_figures.py — Menghasilkan 6 Grafik Publikasi Ilmiah (Figure 1–6) dari hasil simulasi nyata
untuk kelima skenario faktorial (A, B, C1, C2, D) & sensitivitas Latin Hypercube / True PRCC.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SC_COLORS = {
    "A_Baseline_UKNP": "#d62728",
    "B_Managed_UKNP": "#2ca02c",
    "C_Metapop_Uncorrelated": "#1f77b4",
    "C_Metapop_Correlated": "#9467bd",
    "D_Integrated_Managed_Metapop": "#ff7f0e",
}
SC_LABELS = {
    "A_Baseline_UKNP": "A: Ujung Kulon Baseline",
    "B_Managed_UKNP": "B: Ujung Kulon Pengelolaan Intensif (In Situ)",
    "C_Metapop_Uncorrelated": "C1: Metapopulasi Moderat (ρ = 0.0)",
    "C_Metapop_Correlated": "C2: Metapopulasi Moderat (ρ = 0.6)",
    "D_Integrated_Managed_Metapop": "D: Terintegrasi Penuh (Intensif UKNP + Metapop ρ = 0.0)",
}


def make_figure1_conceptual():
    """Figure 1: Diagram konseptual 3 mesin terintegrasi NusantaraRhino."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.4, 3.2, 2.7, 1.35, "#e0f2fe", "#0284c7", "ENGINE 01: DEMOGRAPHY\n(Individual-Based Model)\n• Age, sex, K ceiling\n• 15-16 mo gestation (≥3y IBI)"),
        (0.4, 1.7, 2.7, 1.25, "#dcfce7", "#16a34a", "ENGINE 02: GENETICS\n(20 Diploid Loci + IBD)\n• Drift, Ho/He, AR, F_IBD\n• 1st-yr calf survival exp(-B·F)"),
        (0.4, 0.2, 2.7, 1.25, "#fee2e2", "#dc2626", "ENGINE 03: ENVIRONMENT\n(Chronic + Catastrophes)\n• Annual env noise σ_env\n• Copula P(C1,C2) with ρ"),
        (4.1, 1.7, 2.5, 1.45, "#fef3c7", "#d97706", "INTEGRATED ENGINE\n300 Monte Carlo Runs\n5 Factorial Scenarios\n+ LHS & True PRCC"),
        (7.4, 2.6, 2.3, 1.65, "#ede9fe", "#7c3aed", "SCIENTIFIC OUTPUTS\n• Extinction & 95% Wilson CI\n• He Retention & F_IBD\n• True PRCC & Robustness"),
        (7.4, 0.5, 2.3, 1.65, "#f1f5f9", "#475569", "VALIDATION LAYERS\n• pytest Unit Tests\n• 1970–2020 History\n• Vortex 10 Spec Bridge"),
    ]
    for x, y, w, h, fc, ec, txt in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, lw=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")

    arrows = [
        ((3.18, 3.85), (4.05, 2.7)),
        ((3.18, 2.35), (4.05, 2.4)),
        ((3.18, 0.85), (4.05, 2.0)),
        ((6.68, 2.65), (7.35, 3.35)),
        ((6.68, 2.15), (7.35, 1.4)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2, color="#334155"))

    ax.set_title("Figure 1. Integrated Computational Architecture of NusantaraRhino", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure1_Conceptual_Framework.png", dpi=220)
    plt.savefig(FIG_DIR / "Figure1_Conceptual_Framework.pdf")
    plt.close()


def main():
    sns.set_theme(style="whitegrid")
    make_figure1_conceptual()

    traj = pd.read_csv(TABLE_DIR / "yearly_trajectories_summary.csv")
    wf = pd.read_csv(TABLE_DIR / "wright_fisher_baselines.csv")
    prcc = pd.read_csv(TABLE_DIR / "Table5_Sensitivity_PRCC.csv")
    exp3 = pd.read_csv(TABLE_DIR / "exp3_catastrophe_correlation.csv")
    exp4 = pd.read_csv(TABLE_DIR / "exp4_carrying_capacity.csv")
    exp5 = pd.read_csv(TABLE_DIR / "exp5_combined_robustness.csv")

    # Figure 2: Population Trajectories with P5–P95 Uncertainty Ribbons
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for sc, col in SC_COLORS.items():
        sub = traj[traj["scenario"] == sc]
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["mean_n"], lw=2.4, color=col, label=SC_LABELS[sc])
        ax.fill_between(sub["year"], sub["p5_n"], sub["p95_n"], color=col, alpha=0.12)
    ax.axhline(15, ls="--", color="#333333", lw=1.2, label="Ambang Quasi-Extinction (N = 15)")
    ax.set_title("Figure 2. Lintasan Ukuran Populasi Badak Jawa 100 Tahun (Mean & Interval P5–P95)", fontweight="bold")
    ax.set_xlabel("Tahun Simulasi")
    ax.set_ylabel("Total Populasi Global (Individu)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure2_Population_Trajectories.png", dpi=220)
    plt.savefig(FIG_DIR / "Figure2_Population_Trajectories.pdf")
    plt.close()

    # Figure 3: Extinction & Quasi-Extinction Probability Curves
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    for sc, col in SC_COLORS.items():
        sub = traj[traj["scenario"] == sc]
        if sub.empty:
            continue
        axes[0].plot(sub["year"], sub["p_extinct_yr"], lw=2.3, color=col, label=SC_LABELS[sc])
        axes[1].plot(sub["year"], sub["p_quasi_yr"], lw=2.3, color=col, label=SC_LABELS[sc])
    axes[0].set_title("Figure 3A. Probabilitas Kepunahan Total (N = 0)", fontweight="bold")
    axes[0].set_xlabel("Tahun")
    axes[0].set_ylabel("Probabilitas (%)")
    axes[0].legend(fontsize=7.5)
    axes[1].set_title("Figure 3B. Probabilitas Quasi-Extinction (N ≤ 15)", fontweight="bold")
    axes[1].set_xlabel("Tahun")
    axes[1].set_ylabel("Probabilitas (%)")
    axes[1].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure3_Extinction_Curves.png", dpi=220)
    plt.savefig(FIG_DIR / "Figure3_Extinction_Curves.pdf")
    plt.close()

    # Figure 4: Genetic Diversity Retention (He vs Wright-Fisher) & Inbreeding F_IBD
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    for sc, col in SC_COLORS.items():
        sub = traj[traj["scenario"] == sc]
        if sub.empty:
            continue
        axes[0].plot(sub["year"], sub["mean_he"], lw=2.3, color=col, label=SC_LABELS[sc])
        axes[1].plot(sub["year"], sub["mean_f"], lw=2.3, color=col, label=SC_LABELS[sc])
    axes[0].plot(wf["year"], wf["WF_Ne25"], "k--", lw=1.3, alpha=0.7, label="Teori Wright-Fisher (Ne=25)")
    axes[0].plot(wf["year"], wf["WF_Ne50"], "k:", lw=1.3, alpha=0.7, label="Teori Wright-Fisher (Ne=50)")
    axes[0].set_title("Figure 4A. Retensi Heterozigositas Ekspektasi (He)", fontweight="bold")
    axes[0].set_xlabel("Tahun")
    axes[0].set_ylabel("Expected Heterozygosity (He)")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Figure 4B. Akumulasi Inbreeding Pedigree (F_IBD)", fontweight="bold")
    axes[1].set_xlabel("Tahun")
    axes[1].set_ylabel("Mean F_IBD")
    axes[1].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure4_Genetic_Diversity.png", dpi=220)
    plt.savefig(FIG_DIR / "Figure4_Genetic_Diversity.pdf")
    plt.close()

    # Figure 5: Environmental Catastrophe Correlation & Demographic True PRCC Tornado
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    piv = exp3.pivot(index="Severity", columns="Catastrophe_Corr_rho", values="Quasi_Ext_Prob_Pct")
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "Quasi-Extinction (%)"})
    axes[0].set_title("Figure 5A. Pengaruh Korelasi Bencana (ρ) & Keparahan", fontweight="bold")
    axes[0].set_xlabel("Korelasi Bencana Antar-Lokasi (ρ)")
    axes[0].set_ylabel("Keparahan Bencana (Proporsi Mortalitas)")

    col_prcc = "PRCC_QuasiExt" if "PRCC_QuasiExt" in prcc.columns else "Spearman_r_QuasiExt"
    prcc_plot = prcc.sort_values(col_prcc)
    cols_p = ["#1f77b4" if v < 0 else "#d62728" for v in prcc_plot[col_prcc]]
    axes[1].barh(prcc_plot["Parameter"], prcc_plot[col_prcc], color=cols_p, alpha=0.9)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title("Figure 5B. Sensitivitas Demografi (LHS + True PRCC ke Quasi-Ext.)", fontweight="bold")
    axes[1].set_xlabel("Partial Rank Correlation Coefficient (Biru = Menurunkan Risiko)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure5_Catastrophe_Correlation_Sensitivity.png", dpi=220)
    plt.savefig(FIG_DIR / "Figure5_Catastrophe_Correlation_Sensitivity.pdf")
    plt.close()

    # Figure 6: Combined Robustness Across Uncertainty Regimes & Carrying Capacity K
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    sns.barplot(
        data=exp5, x="Uncertainty_Regime", y="Quasi_Ext_Prob_Pct", hue="Scenario",
        palette=SC_COLORS, ax=axes[0]
    )
    axes[0].set_title("Figure 6A. Ketahanan Strategi di Bawah Ketidakpastian (B & Cat.)", fontweight="bold")
    axes[0].set_xlabel("Rezim Ketidakpastian Parameter")
    axes[0].set_ylabel("Probabilitas Quasi-Extinction (%)")
    axes[0].tick_params(axis="x", labelsize=8.5)
    axes[0].legend(fontsize=7)

    axes[1].plot(exp4["Carrying_Capacity_K"], exp4["Mean_Final_N"], "o-", lw=2.3, color="#2ca02c", label="Mean Final N")
    ax2 = axes[1].twinx()
    ax2.plot(exp4["Carrying_Capacity_K"], exp4["Quasi_Ext_Prob_Pct"], "s--", lw=2.0, color="#d62728", label="Quasi-Ext. (%)")
    axes[1].set_title("Figure 6B. Efek Daya Dukung Habitat Ujung Kulon (K)", fontweight="bold")
    axes[1].set_xlabel("Carrying Capacity (K)")
    axes[1].set_ylabel("Mean Final N (Tahun 100)", color="#2ca02c")
    ax2.set_ylabel("Quasi-Extinction (%)", color="#d62728")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Figure6_Integrated_Robustness.png", dpi=220)
    plt.savefig(FIG_DIR / "Figure6_Integrated_Robustness.pdf")
    plt.close()
    print("Selesai membuat Figure 1–6 (PNG & PDF) di:", FIG_DIR)


if __name__ == "__main__":
    main()
