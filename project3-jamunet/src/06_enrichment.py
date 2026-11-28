"""
06_enrichment.py
================
Langkah 6: Analisis Pengayaan Jalur Biologis (g:Profiler) Target Irisan Jamu–Diabetes.

Tujuannya untuk menjawab:
"Proses seluler dan jalur sinyal apa saja yang paling banyak disentuh oleh gabungan
protein target senyawa jamu yang terkait Diabetes Tipe 2?"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import FIG_DIR, TABLE_DIR, cached_request

GPROF_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def main():
    print("== Langkah 6: Pathway Enrichment Target Jamu–Diabetes ==")
    overlap_df = pd.read_csv(TABLE_DIR / "04_overlapping_targets.csv")
    genes = sorted(overlap_df["gene_symbol"].unique())

    resp = cached_request(
        "gprofiler",
        GPROF_URL,
        method="POST",
        json_body={
            "organism": "hsapiens",
            "query": genes,
            "sources": ["GO:BP", "KEGG", "REAC"],
            "user_threshold": 0.05,
            "significance_threshold_method": "fdr",
        },
    )
    results = (resp or {}).get("result", [])
    df = pd.DataFrame([
        {
            "source": r["source"],
            "term_id": r["native"],
            "term_name": r["name"],
            "p_value": r["p_value"],
            "neg_log10_p": -np.log10(max(r["p_value"], 1e-300)),
            "intersection_size": r["intersection_size"],
            "term_size": r["term_size"],
        }
        for r in results
    ]).sort_values("p_value")
    df.to_csv(TABLE_DIR / "06_pathway_enrichment.csv", index=False)

    print(f"  Total jalur signifikan (FDR < 0.05): {len(df)}")
    print("  Top 10 Jalur (KEGG/REAC/GO:BP):")
    print(df[["source", "term_name", "intersection_size", "p_value"]].head(10).to_string(index=False))

    # Plot dot/bar jalur KEGG + Reactome paling relevan
    kegg_reac = df[df["source"].isin(["KEGG", "REAC"])].head(15).iloc[::-1]
    if kegg_reac.empty:
        kegg_reac = df.head(15).iloc[::-1]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"KEGG": "#d62728", "REAC": "#2ca02c", "GO:BP": "#1f77b4"}
    cols = [palette.get(s, "#666666") for s in kegg_reac["source"]]
    bars = ax.barh(kegg_reac["term_name"], kegg_reac["neg_log10_p"], color=cols, alpha=0.9)
    for bar, cnt in zip(bars, kegg_reac["intersection_size"]):
        ax.text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2, f"n={cnt}", va="center", fontsize=8)
    for src, c in palette.items():
        ax.barh([], [], color=c, label=src)
    ax.legend(title="Sumber")
    ax.set_xlabel("-log10(FDR p-value)")
    ax.set_title("Jalur Biologis Teratas Target Jamu pada Diabetes Tipe 2", fontweight="bold")
    ax.set_xlim(0, kegg_reac["neg_log10_p"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_pathway_enrichment.png", dpi=220)
    plt.close()
    print("Selesai Langkah 6.")


if __name__ == "__main__":
    main()
