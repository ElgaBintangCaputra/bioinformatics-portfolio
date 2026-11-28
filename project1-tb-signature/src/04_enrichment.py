"""
04_enrichment.py
================
Langkah 4: Analisis Pengayaan Jalur Biologis (Pathway & GO Enrichment).

Pertanyaan biologis:
"Kalau ada ratusan gen yang naik saat seseorang sakit TBC aktif, sebenarnya
proses biologis apa yang sedang terjadi di dalam tubuh pasien?"

Kita kirim daftar gen yang naik signifikan ke server g:Profiler
(https://biit.cs.ut.ee/gprofiler/) untuk menguji keterkaitan dengan:
  - GO:BP (Gene Ontology Biological Process)
  - KEGG Pathways
  - REAC (Reactome Pathways)

Hasil API langsung disimpan (cache) ke data/cache/enrichment_gprofiler.json
supaya skrip tetap bisa dijalankan offline tanpa internet di kemudian hari.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_DIR / "data" / "cache"
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def query_gprofiler(gene_list: list[str], background: list[str], cache_file: Path) -> list[dict]:
    """Kirim kueri ke g:Profiler REST API dengan cache lokal."""
    if cache_file.exists():
        print(f"  [cache] Membaca hasil g:Profiler dari {cache_file.name}")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": "hsapiens",
        "query": gene_list,
        "sources": ["GO:BP", "KEGG", "REAC"],
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "domain_scope": "custom",
        "background": background,
        "no_evidences": False,
    }
    print(f"  [API] Mengirim {len(gene_list)} gen ke g:Profiler...")
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    results = resp.json().get("result", [])
    cache_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    print("== Langkah 4: Pathway Enrichment Analysis ==")
    de = pd.read_csv(TABLE_DIR / "03_differential_expression.csv")
    up_genes = de.loc[de["status"] == "Naik di TB (Up)", "gene"].tolist()
    bg_genes = de["gene"].tolist()
    print(f"  Jumlah gen 'Naik di TB': {len(up_genes)} (latar belakang: {len(bg_genes)} gen)")

    results = query_gprofiler(up_genes, bg_genes, CACHE_DIR / "enrichment_gprofiler.json")
    if not results:
        print("  Tidak ada jalur signifikan ditemukan.")
        return

    df = pd.DataFrame([
        {
            "source": r["source"],
            "term_id": r["native"],
            "term_name": r["name"],
            "p_value": r["p_value"],
            "neg_log10_p": -np.log10(max(r["p_value"], 1e-300)),
            "term_size": r["term_size"],
            "query_size": r["query_size"],
            "intersection_size": r["intersection_size"],
            "precision": r["precision"],
        }
        for r in results
    ]).sort_values("p_value")

    df.to_csv(TABLE_DIR / "04_pathway_enrichment.csv", index=False)
    print(f"  Total jalur biologis signifikan (FDR < 0.05): {len(df)}")
    print("  Top 10 Jalur Biologis:")
    print(df[["source", "term_name", "intersection_size", "p_value"]].head(10).to_string(index=False))

    # Plot top 15 jalur (campuran GO:BP, KEGG, Reactome)
    top15 = df.head(15).iloc[::-1]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = {"GO:BP": "#1f77b4", "KEGG": "#d62728", "REAC": "#2ca02c"}
    bar_colors = [palette.get(s, "#7f7f7f") for s in top15["source"]]

    bars = ax.barh(top15["term_name"], top15["neg_log10_p"], color=bar_colors, alpha=0.9)
    for bar, count in zip(bars, top15["intersection_size"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"n={count}", va="center", fontsize=8, color="#333333"
        )

    # Legenda sumber
    for src, col in palette.items():
        ax.barh([], [], color=col, label=src)
    ax.legend(title="Basis Data", loc="lower right")
    ax.set_xlabel("-log10(FDR p-value)")
    ax.set_title("Top 15 Jalur Biologis yang Aktif pada Pasien TBC", fontsize=12, fontweight="bold")
    ax.set_xlim(0, top15["neg_log10_p"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_pathway_enrichment.png", dpi=220)
    plt.close()
    print("Selesai! Grafik pengayaan jalur tersimpan.")


if __name__ == "__main__":
    main()
