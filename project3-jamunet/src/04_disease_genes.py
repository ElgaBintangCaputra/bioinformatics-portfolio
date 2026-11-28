"""
04_disease_genes.py
===================
Langkah 4: Mengambil Gen Terkait Diabetes Melitus Tipe 2 (Open Targets Platform) & Irisannya.

Alur:
1. Kirim kueri GraphQL ke Open Targets Platform API v4 untuk penyakit
   `MONDO_0005148` (Type 2 diabetes mellitus).
2. Ambil hingga 1.000 gen teratas beserta skor asosiasi keseluruhan (`score`).
3. Irisan antara:
   - Himpunan A = target protein manusia dari senyawa jamu (ChEMBL)
   - Himpunan B = gen terkait Diabetes Tipe 2 (Open Targets)
   -> Himpunan (A ∩ B) adalah **kandidat target kerja jamu pada Diabetes Tipe 2**!
4. Gambar diagram Venn / irisan yang rapi dan simpan tabel hasil.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

from common import FIG_DIR, T2D_ID, TABLE_DIR, cached_request

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

QUERY = """
query T2DTargets($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: { index: $index, size: $size }) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
      }
    }
  }
}
"""


def fetch_t2d_targets(max_pages: int = 2, page_size: int = 500) -> pd.DataFrame:
    rows = []
    for page_idx in range(max_pages):
        resp = cached_request(
            "opentargets",
            OT_URL,
            method="POST",
            json_body={
                "query": QUERY,
                "variables": {"efoId": T2D_ID, "index": page_idx, "size": page_size},
            },
        )
        disease = (resp or {}).get("data", {}).get("disease")
        if not disease:
            break
        batch = disease["associatedTargets"]["rows"]
        if not batch:
            break
        for r in batch:
            rows.append({
                "ensembl_id": r["target"]["id"],
                "gene_symbol": r["target"]["approvedSymbol"],
                "gene_name": r["target"]["approvedName"],
                "ot_score": round(float(r["score"]), 4),
            })
    return pd.DataFrame(rows).drop_duplicates("gene_symbol")


def main():
    print("== Langkah 4: Penelusuran Gen Diabetes Tipe 2 di Open Targets ==")
    t2d_df = fetch_t2d_targets(max_pages=6, page_size=500)
    t2d_df = t2d_df[t2d_df["ot_score"] >= 0.02].copy()
    t2d_df.to_csv(TABLE_DIR / "04_t2d_genes.csv", index=False)

    ct_df = pd.read_csv(TABLE_DIR / "03_compound_targets.csv")
    jamu_genes = set(ct_df["gene_symbol"].dropna())
    t2d_genes = set(t2d_df["gene_symbol"].dropna())
    overlap = sorted(jamu_genes & t2d_genes)

    overlap_df = (
        ct_df[ct_df["gene_symbol"].isin(overlap)]
        .groupby("gene_symbol", as_index=False)
        .agg(
            n_compounds=("compound", "nunique"),
            compounds=("compound", lambda s: "; ".join(sorted(set(s)))),
            plants=("plants", lambda s: "; ".join(sorted(set(p.strip() for x in s for p in x.split(";"))))),
            max_pchembl=("max_pchembl", "max"),
        )
        .merge(t2d_df, on="gene_symbol", how="left")
        .sort_values(["n_compounds", "ot_score"], ascending=[False, False])
    )
    overlap_df.to_csv(TABLE_DIR / "04_overlapping_targets.csv", index=False)

    print(f"  Gen target senyawa jamu (ChEMBL): {len(jamu_genes)}")
    print(f"  Gen terkait T2D diambil (Open Targets, score >= 0.02): {len(t2d_genes)}")
    print(f"  Irisan (Target Jamu ∩ Diabetes Tipe 2): {len(overlap)} gen")
    print("  Top 10 gen irisan:")
    print(overlap_df[["gene_symbol", "n_compounds", "ot_score", "max_pchembl"]].head(10).to_string(index=False))

    # --- Plot Diagram Venn Sederhana ---
    fig, ax = plt.subplots(figsize=(8, 5.2))
    c1 = Circle((-0.7, 0), 1.35, facecolor="#2ca02c", alpha=0.35, edgecolor="#1b6e1b", lw=2)
    c2 = Circle((0.7, 0), 1.35, facecolor="#1f77b4", alpha=0.35, edgecolor="#104e8b", lw=2)
    ax.add_patch(c1)
    ax.add_patch(c2)

    ax.text(-1.25, 0.1, f"{len(jamu_genes - t2d_genes)}\nTarget\nJamu Saja", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(1.25, 0.1, f"{len(t2d_genes - jamu_genes)}\nGen T2D\nLainnya", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.0, 0.1, f"{len(overlap)}\nIrisan\n(T2D ∩ Jamu)", ha="center", va="center", fontsize=13, fontweight="bold", color="#b22222")

    ax.text(-0.9, 1.55, f"Target ChEMBL Jamu (n={len(jamu_genes)})", ha="center", fontsize=11, fontweight="bold", color="#1b6e1b")
    ax.text(0.9, 1.55, f"Open Targets T2D (n={len(t2d_genes)})", ha="center", fontsize=11, fontweight="bold", color="#104e8b")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.7, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Irisan Target Protein Senyawa Jamu dengan Gen Diabetes Tipe 2", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_target_t2d_overlap.png", dpi=220)
    plt.close()
    print("Selesai Langkah 4.")


if __name__ == "__main__":
    main()
