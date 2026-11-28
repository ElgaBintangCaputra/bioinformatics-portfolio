"""
02_preprocess.py
================
Langkah 2: Membersihkan dan menormalisasi data ekspresi gen dari GEO.

Apa yang dilakukan skrip ini?
1. Membaca file Series Matrix (*.txt.gz) untuk:
   - Discovery cohort (GPL6947): GSE19444, GSE19439, GSE19442
   - External validation (GPL10558): GSE37250
2. Mengekstrak label klinis setiap sampel (Active TB vs Latent TB / Healthy / Other Disease).
3. Mengecek skala nilai ekspresi:
   - Nilai microarray Illumina di GEO ini masih berskala linear (ratusan - puluhan ribu).
   - Kita lakukan transformasi log2(max(x, 1.0)) agar distribusinya simetris.
   - Lalu kita lakukan Quantile Normalization di dalam masing-masing dataset supaya
     distribusi antar-sampel sejajar.
4. Memetakan ID probe ("ILMN_...") ke simbol gen resmi ("GBP1", "FCGR1A", dll.)
   memakai file anotasi platform (*.annot.gz). Jika satu gen punya beberapa probe,
   kita pilih probe dengan rata-rata ekspresi tertinggi (praktik standar microarray).
5. Menyelaraskan daftar gen yang sama antara GPL6947 (discovery) dan GPL10558 (validasi),
   lalu menyaring gen dengan varians sangat rendah di discovery set.
6. Menghasilkan grafik QC (Quality Control): distribusi sampel & plot PCA.
"""

import gzip
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROC_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"

for d in (PROC_DIR, FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def parse_series_matrix(gz_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membaca file *_series_matrix.txt.gz dari NCBI GEO.
    Mengembalikan:
      - meta: DataFrame metadata per sampel (index = GSM ID)
      - expr: DataFrame ekspresi (index = ID probe, kolom = GSM ID)
    """
    sample_ids = []
    sources = []
    titles = []
    char_rows = []
    table_lines = []
    in_table = False

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip('"') for x in line.rstrip("\r\n").split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\r\n").split("\t")[1:]]
            elif line.startswith("!Sample_source_name_ch1"):
                sources = [x.strip('"') for x in line.rstrip("\r\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip('"') for x in line.rstrip("\r\n").split("\t")[1:]]
                char_rows.append(vals)
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                table_lines.append(line)

    # Susun tabel ekspresi menggunakan pandas dari string buffer
    expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [str(c).strip('"') for c in expr.columns]

    # Susun metadata sampel
    meta = pd.DataFrame({"sample_id": sample_ids, "title": titles, "source": sources})
    for idx, row in enumerate(char_rows):
        if len(row) == len(sample_ids):
            meta[f"char_{idx}"] = row
    meta = meta.set_index("sample_id")
    return meta, expr


def standarkan_label(meta: pd.DataFrame, gse_name: str) -> pd.DataFrame:
    """
    Mengubah deskripsi teks dari GEO menjadi label kelas yang seragam:
      - 'Active_TB'
      - 'Latent_TB'
      - 'Healthy'
      - 'Other_Disease'
    Serta label biner: y = 1 (Active_TB), y = 0 (Non-TB).

    Catatan: gunakan kolom `source` (Sample_source_name_ch1) untuk status utama
    supaya tidak tertipu oleh nama kolom lain seperti `exposure_latent: n.a`.
    """
    src = meta["source"].fillna("").astype(str).str.lower()
    gabungan = meta.fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

    status = []
    for teks in src:
        if "active tb" in teks or "active tuberculosis" in teks:
            status.append("Active_TB")
        elif "latent" in teks or "ltbi" in teks:
            status.append("Latent_TB")
        elif "healthy" in teks or "control" in teks:
            status.append("Healthy")
        elif "other disease" in teks:
            status.append("Other_Disease")
        else:
            status.append("Unknown")

    out = pd.DataFrame(index=meta.index)
    out["dataset"] = gse_name
    out["group"] = status
    out["y_tb"] = (out["group"] == "Active_TB").astype(int)

    # Ekstrak status HIV jika ada (penting untuk analisis subgrup GSE37250)
    hiv = []
    for teks in gabungan:
        if "hiv positive" in teks:
            hiv.append("HIV+")
        elif "hiv negative" in teks:
            hiv.append("HIV-")
        else:
            hiv.append("Unknown")
    out["hiv_status"] = hiv
    return out


def baca_anotasi_platform(annot_path: Path) -> pd.Series:
    """
    Membaca file GPL*.annot.gz dan mengembalikan Series: index = ID probe, value = Gene symbol.
    Jika satu probe dipetakan ke beberapa gen ("GENE1///GENE2"), ambil yang pertama saja.
    """
    lines = []
    in_table = False
    with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
            elif line.startswith("!platform_table_end"):
                break
            elif in_table:
                lines.append(line)

    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", low_memory=False)
    df["ID"] = df["ID"].astype(str).str.strip()
    df["Gene symbol"] = df["Gene symbol"].fillna("").astype(str).str.strip()
    # Bersihkan multi-mapping (ambil simbol pertama)
    df["gene"] = df["Gene symbol"].str.split("///").str[0].str.strip()
    df = df[df["gene"] != ""]
    return df.set_index("ID")["gene"]


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile Normalization (normalisasi kuantil):
    Menyamakan distribusi intensitas di setiap sampel (kolom) tanpa mengubah
    urutan peringkat gen di dalam sampel tersebut. Standar emas untuk microarray!
    """
    sorted_vals = np.sort(df.values, axis=0)
    mean_dist = np.mean(sorted_vals, axis=1)
    # Ganti setiap nilai dengan rata-rata pada peringkatnya
    ranks = np.argsort(np.argsort(df.values, axis=0), axis=0)
    normalized = mean_dist[ranks]
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def probe_ke_gen(expr: pd.DataFrame, probe_map: pd.Series) -> pd.DataFrame:
    """
    Memetakan matriks probe x sampel menjadi gen x sampel.
    Jika satu gen punya >1 probe, pilih probe dengan ekspresi rata-rata tertinggi.
    """
    common = expr.index.intersection(probe_map.index)
    sub = expr.loc[common].copy()
    sub["gene"] = probe_map.loc[common].values
    sub["mean_expr"] = sub.drop(columns="gene").mean(axis=1)
    # Urutkan dari ekspresi tertinggi lalu buang duplikat nama gen
    sub = sub.sort_values("mean_expr", ascending=False).drop_duplicates(subset="gene")
    sub = sub.set_index("gene").drop(columns="mean_expr")
    return sub


def proses_kumpulan(gse_list: list[str], probe_map: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gabungkan beberapa GSE (pada platform yang sama), log2, quantile norm, dan petakan ke gen."""
    expr_parts = []
    meta_parts = []

    for gse in gse_list:
        print(f"  Membaca {gse}...")
        meta_raw, expr_raw = parse_series_matrix(RAW_DIR / f"{gse}_series_matrix.txt.gz")
        meta_std = standarkan_label(meta_raw, gse)

        # Pastikan urutan kolom sama dengan metadata dan buang label Unknown jika ada
        valid = meta_std[meta_std["group"] != "Unknown"].index
        meta_std = meta_std.loc[valid]
        expr_raw = expr_raw[valid].apply(pd.to_numeric, errors="coerce").dropna()

        # Cek apakah perlu log2 (jika median > 50 berarti masih skala linear)
        if expr_raw.values.mean() > 50:
            expr_log = np.log2(np.clip(expr_raw, 1.0, None))
        else:
            expr_log = expr_raw

        expr_parts.append(expr_log)
        meta_parts.append(meta_std)

    # Gabungkan kolom-kolom sampel dari seluruh GSE dalam kohort ini
    common_probes = expr_parts[0].index
    for part in expr_parts[1:]:
        common_probes = common_probes.intersection(part.index)
    expr_all = pd.concat([p.loc[common_probes] for p in expr_parts], axis=1)
    meta_all = pd.concat(meta_parts, axis=0)

    # Quantile normalization agar perbedaan batch antar-sampel lebih rata
    expr_qn = quantile_normalize(expr_all)
    expr_gene = probe_ke_gen(expr_qn, probe_map)
    return expr_gene, meta_all


def buat_plot_qc(expr_disc: pd.DataFrame, meta_disc: pd.DataFrame,
                 expr_val: pd.DataFrame, meta_val: pd.DataFrame) -> None:
    """Membuat grafik QC: PCA Discovery + Distribusi intensitas sampel."""
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. PCA Discovery Cohort
    X = expr_disc.T.values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    pca_df = meta_disc.copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df.to_csv(TABLE_DIR / "02_pca_discovery.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"Active_TB": "#d62728", "Latent_TB": "#ff7f0e", "Healthy": "#2ca02c"}
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="group", style="dataset",
        palette=palette, s=75, alpha=0.88, ax=axes[0]
    )
    axes[0].set_title(f"PCA Discovery Cohort (n={len(pca_df)})", fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({var[0]:.1f}% varians)")
    axes[0].set_ylabel(f"PC2 ({var[1]:.1f}% varians)")
    axes[0].legend(fontsize=8, loc="best")

    # 2. PCA External Validation Cohort
    Xv = StandardScaler().fit_transform(expr_val.T.values)
    coords_v = PCA(n_components=2, random_state=42).fit_transform(Xv)
    var_v = PCA(n_components=2, random_state=42).fit(Xv).explained_variance_ratio_ * 100
    pv_df = meta_val.copy()
    pv_df["PC1"] = coords_v[:, 0]
    pv_df["PC2"] = coords_v[:, 1]
    palette_v = {
        "Active_TB": "#d62728",
        "Latent_TB": "#ff7f0e",
        "Other_Disease": "#7f7f7f",
    }
    sns.scatterplot(
        data=pv_df, x="PC1", y="PC2", hue="group",
        palette=palette_v, s=45, alpha=0.75, ax=axes[1]
    )
    axes[1].set_title(f"PCA External Validation GSE37250 (n={len(pv_df)})", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({var_v[0]:.1f}% varians)")
    axes[1].set_ylabel(f"PC2 ({var_v[1]:.1f}% varians)")
    axes[1].legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_qc_pca.png", dpi=200)
    plt.close()


def main():
    print("== Langkah 2: Preprocessing & Normalisasi ==")
    map_gpl6947 = baca_anotasi_platform(RAW_DIR / "GPL6947.annot.gz")
    map_gpl10558 = baca_anotasi_platform(RAW_DIR / "GPL10558.annot.gz")
    print(f"  Probe terpetakan GPL6947: {len(map_gpl6947)}, GPL10558: {len(map_gpl10558)}")

    print("-> Memproses Discovery Cohort (GSE19444, GSE19439, GSE19442)...")
    expr_disc, meta_disc = proses_kumpulan(["GSE19444", "GSE19439", "GSE19442"], map_gpl6947)

    print("-> Memproses External Validation Cohort (GSE37250)...")
    expr_val, meta_val = proses_kumpulan(["GSE37250"], map_gpl10558)

    # Ambil hanya gen yang hadir di kedua platform agar model ML bisa langsung diuji lintas platform
    common_genes = sorted(expr_disc.index.intersection(expr_val.index))
    expr_disc = expr_disc.loc[common_genes]
    expr_val = expr_val.loc[common_genes]
    print(f"  Gen bersama antar platform: {len(common_genes)}")

    # Simpan ekspresi panel literatur Sweeney 3-gene (GBP5, DUSP3, KLF2) SEBELUM filter varians
    # agar benchmark eksternal di Langkah 7 menggunakan ketiga gen Sweeney secara utuh!
    sweeney_genes = [g for g in ["GBP5", "DUSP3", "KLF2"] if g in expr_disc.index]
    expr_disc.loc[sweeney_genes].to_parquet(PROC_DIR / "sweeney3_disc_expr.parquet")
    expr_val.loc[sweeney_genes].to_parquet(PROC_DIR / "sweeney3_val_expr.parquet")
    print(f"  Panel literatur Sweeney tersimpan utuh ({len(sweeney_genes)}/3 gen): {sweeney_genes}")

    # Saring 50% gen dengan varians tertinggi di Discovery (mengurangi noise tanpa melihat label/val)
    variances = expr_disc.var(axis=1)
    top_var_genes = variances.nlargest(int(len(variances) * 0.5)).index
    expr_disc = expr_disc.loc[top_var_genes]
    expr_val = expr_val.loc[top_var_genes]
    print(f"  Setelah filter varians (top 50% di discovery): {len(top_var_genes)} gen")

    # Simpan ke folder processed
    expr_disc.to_parquet(PROC_DIR / "discovery_expr.parquet")
    meta_disc.to_csv(PROC_DIR / "discovery_meta.csv")
    expr_val.to_parquet(PROC_DIR / "validation_expr.parquet")
    meta_val.to_csv(PROC_DIR / "validation_meta.csv")

    # Simpan ringkasan jumlah sampel per kelompok
    ringkasan = pd.concat([
        meta_disc.groupby(["dataset", "group"]).size().rename("jumlah_sampel"),
        meta_val.groupby(["dataset", "group"]).size().rename("jumlah_sampel"),
    ]).reset_index()
    ringkasan.to_csv(TABLE_DIR / "02_sample_summary.csv", index=False)
    print("\nRingkasan Sampel:")
    print(ringkasan.to_string(index=False))

    buat_plot_qc(expr_disc, meta_disc, expr_val, meta_val)
    print("Selesai! Data bersih tersimpan di:", PROC_DIR)


if __name__ == "__main__":
    main()
