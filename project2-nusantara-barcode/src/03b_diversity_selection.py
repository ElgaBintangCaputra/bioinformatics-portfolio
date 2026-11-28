"""
03b_diversity_selection.py
==========================
Langkah 3b (Pendalaman Genetika Populasi, Verifikasi ORF Mitokondria, & Audit Presisi GenBank):
1. Menghitung statistik genetika populasi klasik per spesies ikan:
   - Jumlah haplotipe (h) & Keragaman haplotipe (Hd)
   - Keragaman nukleotida (pi) & Situs polimorfik (S)
   - Tajima's D (uji netralitas evolusi / ekspansi populasi)
2. Verifikasi Open Reading Frame (ORF) Kode Genetik Mitokondria Vertebrata (Translasi Tabel 2 NCBI)
   & Analisis variasi per posisi kodon (Posisi 1, 2, vs 3) serta Sliding Window pi(x)
   di sepanjang 636 bp pada SELURUH 229 sekuens COI (tanpa bias sub-sampling!).
3. Audit otomatis kesalahan label (Asymmetric Precision Misannotation Detector) pada catatan GenBank:
   - Membedakan spesimen outlier yang benar-benar salah label (mis. PQ813541.1) dari
     klaster referensi bersih yang menjadi tetangganya (sehingga referensi bersih tidak ikut tertuduh!).
   - Membedakan introgresi mitokondria kongenerik (Thunnus albacares vs T. obesus) dari
     divergensi kriptik dalam genus dan kontaminasi lintas-famili.
4. Uji Monofili pada pohon Maximum-Likelihood IQ-TREE 2 (`iqtree_full.treefile`).
"""

import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import AlignIO, Phylo
from Bio.Seq import Seq

from common import PROJECT_DIR, get_paths, load_config, setup_logger

VALID = {"A", "C", "G", "T"}


def tajimas_d(n: int, S: int, k_hat: float) -> float:
    """Menghitung Tajima's D dari jumlah sampel n, situs segregasi S, dan rata-rata selisih berpasangan k_hat."""
    if n < 4 or S == 0:
        return np.nan
    a1 = sum(1.0 / i for i in range(1, n))
    a2 = sum(1.0 / (i * i) for i in range(1, n))
    b1 = (n + 1.0) / (3.0 * (n - 1.0))
    b2 = (2.0 * (n * n + n + 3.0)) / (9.0 * n * (n - 1.0))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2.0) / (a1 * n) + a2 / (a1 * a1)
    e1 = c1 / a1
    e2 = c2 / (a1 * a1 + a2)
    denom = math.sqrt(max(1e-12, e1 * S + e2 * S * (S - 1)))
    theta_w = S / a1
    return (k_hat - theta_w) / denom


def statistik_populasi(seqs: list[str]) -> dict:
    n = len(seqs)
    L = len(seqs[0])
    counts = Counter(seqs)
    h = len(counts)
    hd = (n / (n - 1.0)) * (1.0 - sum((c / n) ** 2 for c in counts.values())) if n > 1 else 0.0

    S = 0
    for col in range(L):
        bases = {s[col] for s in seqs if s[col] in VALID}
        if len(bases) > 1:
            S += 1

    diffs = []
    valid_lens = []
    for i in range(n):
        for j in range(i + 1, n):
            d = 0
            v = 0
            for a, b in zip(seqs[i], seqs[j]):
                if a in VALID and b in VALID:
                    v += 1
                    if a != b:
                        d += 1
            diffs.append(d)
            valid_lens.append(v)

    k_hat = float(np.mean(diffs)) if diffs else 0.0
    pi = float(np.mean([d / v for d, v in zip(diffs, valid_lens) if v > 0])) if diffs else 0.0
    td = tajimas_d(n, S, k_hat)
    return {
        "h": h,
        "Hd": round(hd, 4),
        "S": S,
        "pi": round(pi, 5),
        "tajimas_D": round(td, 3) if not np.isnan(td) else np.nan,
    }


def verifikasi_orf_mitokondria(seqs: list[str], table_id: int = 2) -> tuple[int, dict[int, int]]:
    """
    Mengecek ketiga kerangka baca (reading frame 0, 1, 2) menggunakan Kode Genetik
    Mitokondria Vertebrata (NCBI Translation Table 2) untuk menemukan frame fungsional
    bebas stop codon internal (uji bebas pseudogen NUMT).
    """
    stop_counts = {}
    for frame in (0, 1, 2):
        total_stops = 0
        for s in seqs:
            clean = "".join(c if c in VALID else "N" for c in s[frame:])
            trim_len = (len(clean) // 3) * 3
            if trim_len < 3:
                continue
            aa = str(Seq(clean[:trim_len]).translate(table=table_id))
            # Hitung stop codon internal (kecuali kodon terakhir)
            total_stops += aa[:-1].count("*")
        stop_counts[frame] = total_stops
    best_frame = min(stop_counts, key=stop_counts.get)
    return best_frame, stop_counts


def analisis_posisi_kodon(seqs: list[str]) -> pd.DataFrame:
    """Bandingkan GC%, % situs polimorfik, dan verifikasi ORF pada posisi kodon ke-1, ke-2, dan ke-3."""
    L = len(seqs[0])
    best_frame, stop_counts = verifikasi_orf_mitokondria(seqs, table_id=2)
    rows = []
    for pos in (0, 1, 2):
        col_idx = (best_frame + pos) % 3
        cols = list(range(col_idx, L, 3))
        poly = 0
        gc = 0
        tot = 0
        for c in cols:
            bases = [s[c] for s in seqs if s[c] in VALID]
            if len(set(bases)) > 1:
                poly += 1
            gc += sum(1 for b in bases if b in ("G", "C"))
            tot += len(bases)
        rows.append({
            "Posisi_Kodon": f"Posisi {pos + 1}",
            "Jumlah_Situs": len(cols),
            "Situs_Polimorfik": poly,
            "Persen_Polimorfik": round(100 * poly / max(1, len(cols)), 1),
            "GC_Persen": round(100 * gc / max(1, tot), 1),
            "ORF_Frame_Terpilih": best_frame + 1,
            "Total_Internal_Stop_ORF": stop_counts[best_frame],
        })
    return pd.DataFrame(rows)


def hitung_sliding_window_pi_penuh(seqs: list[str], win: int = 45, step: int = 15) -> tuple[list[float], list[float]]:
    """
    Menghitung keragaman nukleotida per situs pi(x) dengan koreksi ukuran sampel n/(n-1)
    pada SELURUH sekuens (tanpa pemotongan 80 sekuens pertama).
    """
    L = len(seqs[0])
    site_pi = np.zeros(L, dtype=float)
    for c in range(L):
        col_b = [s[c] for s in seqs if s[c] in VALID]
        n_b = len(col_b)
        if n_b <= 1:
            site_pi[c] = 0.0
            continue
        freqs = Counter(col_b)
        sum_sq = sum((cnt / n_b) ** 2 for cnt in freqs.values())
        site_pi[c] = (n_b / (n_b - 1.0)) * (1.0 - sum_sq)

    win_centers, win_pi = [], []
    for start in range(0, L - win + 1, step):
        win_centers.append(start + win / 2.0)
        win_pi.append(float(np.mean(site_pi[start : start + win])))
    return win_centers, win_pi


def audit_misannotation(dist_csv: Path, threshold: float = 0.05) -> pd.DataFrame:
    """
    Deteksi otomatis asimetris (Precision GenBank Audit):
    - Tidak menuduh klaster referensi yang bersih (mean_intra <= 0.02) ketika ada 1 spesimen
      outlier dari spesies/famili lain yang salah label dan identik dengannya.
    - Membedakan:
      1) Suspek Kuat Mislabel Spesimen (Identik Spesies Lain) -> Exclude_From_Reference
      2) Outlier Lintas-Famili / Suspek Mislabel -> Exclude_From_Reference
      3) Introgresi / Divergensi Dangkal Kongenerik (Thunnus) -> Require_Nuclear_Marker
      4) Divergensi Intraspesifik Tinggi (Kriptik / Kompleks Spesies) -> Review_Taxonomy
    """
    df = pd.read_csv(dist_csv, index_col=0)
    ids = df.index.tolist()
    sp = [i.split("|")[1].replace("_", " ") for i in ids]
    genus = [s.split(" ")[0] for s in sp]
    fam = [i.split("|")[2] for i in ids]

    # Langkah 1: Hitung mean_intra masing-masing sekuens terlebih dahulu
    mean_intra_all = np.zeros(len(ids), dtype=float)
    for idx in range(len(ids)):
        same_idx = [j for j, s in enumerate(sp) if s == sp[idx] and j != idx]
        mean_intra_all[idx] = float(df.iloc[idx, same_idx].mean()) if same_idx else 0.0

    flagged = []
    for idx, seq_id in enumerate(ids):
        own_sp = sp[idx]
        own_gen = genus[idx]
        own_fam = fam[idx]
        mean_intra = mean_intra_all[idx]

        other_idx = [j for j, s in enumerate(sp) if s != own_sp]
        nearest_other_pos = other_idx[int(np.argmin(df.iloc[idx, other_idx].values))]
        min_inter = float(df.iloc[idx, nearest_other_pos])
        other_sp = sp[nearest_other_pos]
        other_gen = genus[nearest_other_pos]
        other_fam = fam[nearest_other_pos]
        other_mean_intra = mean_intra_all[nearest_other_pos]

        if mean_intra <= threshold and min_inter >= mean_intra:
            continue

        # Cek Asimetri: Jika sekuens ini sangat kompak di spesiesnya sendiri (mean_intra <= 0.02),
        # berbeda genus dengan tetangga terdekat, dan tetangga terdekat itu sendiri adalah outlier
        # parah di spesies asalnya (other_mean_intra > threshold), maka sekuens ini adalah
        # REFERENSI BERSIH yang hanya didekati oleh spesimen salah label! Jangan tuduh sekuens ini.
        if mean_intra <= 0.02 and own_gen != other_gen and other_mean_intra > threshold:
            continue

        if min_inter < 0.02 and mean_intra > threshold:
            status = "Suspek Kuat Mislabel Spesimen (Identik Spesies Lain)"
            action = "Exclude_From_Reference"
        elif mean_intra > threshold and own_fam != other_fam:
            status = "Outlier Lintas-Famili / Suspek Mislabel"
            action = "Exclude_From_Reference"
        elif min_inter < 0.02 and own_gen == other_gen:
            status = f"Introgresi / Divergensi Dangkal Kongenerik ({own_gen})"
            action = "Require_Nuclear_Marker"
        else:
            status = "Divergensi Intraspesifik Tinggi (Kriptik / Kompleks Spesies)"
            action = "Review_Taxonomy"

        flagged.append({
            "seq_id": seq_id,
            "accession": seq_id.split("|")[0],
            "nominal_species": own_sp,
            "nominal_family": own_fam,
            "mean_dist_to_own_species": round(mean_intra, 4),
            "nearest_other_seq": ids[nearest_other_pos],
            "nearest_other_species": other_sp,
            "min_dist_to_other": round(min_inter, 4),
            "status_audit": status,
            "action_recommended": action,
        })
    return pd.DataFrame(flagged)


def cek_monofili_pohon(treefile: Path) -> pd.DataFrame:
    """Uji apakah tiap spesies membentuk klad monofiletik di pohon IQ-TREE 2."""
    if not treefile.exists():
        return pd.DataFrame()
    tree = Phylo.read(treefile, "newick")
    tips = tree.get_terminals()
    by_sp = {}
    for t in tips:
        sp = t.name.split("|")[1].replace("_", " ")
        by_sp.setdefault(sp, []).append(t)

    rows = []
    for sp, sp_tips in sorted(by_sp.items()):
        mrca = tree.common_ancestor(sp_tips)
        mrca_tips = mrca.get_terminals()
        is_mono = len(mrca_tips) == len(sp_tips)
        rows.append({
            "species": sp,
            "n_tips": len(sp_tips),
            "n_tips_in_mrca": len(mrca_tips),
            "monophyletic": is_mono,
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    logger.info("== Langkah 3b: Genetika Populasi, Verifikasi ORF, Audit Presisi GenBank & Monofili ==")

    aln = AlignIO.read(paths["processed_dir"] / "coi_aligned_trimmed.fasta", "fasta")
    seqs = [str(r.seq).upper() for r in aln]
    ids = [r.id for r in aln]
    by_sp = {}
    for rid, s in zip(ids, seqs):
        sp = rid.split("|")[1].replace("_", " ")
        by_sp.setdefault(sp, []).append(s)

    pop_df = pd.DataFrame([
        {"species": sp, "n": len(ss), **statistik_populasi(ss)} for sp, ss in sorted(by_sp.items())
    ])
    pop_df.to_csv(paths["tables_dir"] / "03b_population_genetics.csv", index=False)
    logger.info(
        f"  Rata-rata pi antar-individu per spesies: {pop_df['pi'].mean():.5f} | Rata-rata Hd: {pop_df['Hd'].mean():.3f}"
    )

    codon_df = analisis_posisi_kodon(seqs)
    codon_df.to_csv(paths["tables_dir"] / "03b_codon_position_stats.csv", index=False)
    logger.info("  Statistik Posisi Kodon & Verifikasi ORF Mitokondria:\n" + codon_df.to_string(index=False))

    audit_df = audit_misannotation(paths["tables_dir"] / "03_k2p_distance_matrix.csv")
    audit_df.to_csv(paths["tables_dir"] / "03b_genbank_audit.csv", index=False)
    n_excl = int((audit_df["action_recommended"] == "Exclude_From_Reference").sum()) if not audit_df.empty else 0
    logger.info(
        f"  Audit Presisi GenBank: {len(audit_df)} sekuens anomali terdeteksi "
        f"({n_excl} spesimen salah label/outlier lintas-famili direkomendasikan Exclude_From_Reference)."
    )
    if not audit_df.empty:
        logger.info("\n" + audit_df.to_string(index=False))

    mono_df = cek_monofili_pohon(PROJECT_DIR / "results" / "trees" / "iqtree_full.treefile")
    if not mono_df.empty:
        mono_df.to_csv(paths["tables_dir"] / "03b_monophyly_check.csv", index=False)
        n_mono = int(mono_df["monophyletic"].sum())
        logger.info(f"  Uji Monofili Pohon IQ-TREE: {n_mono}/{len(mono_df)} spesies membentuk klad monofiletik murni.")

    # Sliding window keragaman nukleotida sepanjang 636 bp COI pada SELURUH 229 sekuens
    win_centers, win_pi = hitung_sliding_window_pi_penuh(seqs, win=45, step=15)

    # Plot 2 panel: (Kiri) Sliding Window Variabilitas COI (n=229), (Kanan) Posisi Kodon 1 vs 2 vs 3
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(win_centers, win_pi, lw=2.3, color="#1f77b4")
    axes[0].fill_between(win_centers, win_pi, alpha=0.2, color="#1f77b4")
    axes[0].set_title(f"Profil Sliding-Window Keragaman Nukleotida COI (636 bp, n={len(seqs)})", fontweight="bold")
    axes[0].set_xlabel("Posisi Nukleotida (bp)")
    axes[0].set_ylabel("Keragaman Nukleotida (π per Situs)")

    x = np.arange(len(codon_df))
    w = 0.35
    axes[1].bar(x - w / 2, codon_df["Persen_Polimorfik"], width=w, color="#d62728", label="% Situs Polimorfik")
    axes[1].bar(x + w / 2, codon_df["GC_Persen"], width=w, color="#2ca02c", label="GC %")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(codon_df["Posisi_Kodon"])
    axes[1].set_title("Variabilitas & GC% per Posisi Kodon Gen COI (ORF Terverifikasi)", fontweight="bold")
    axes[1].set_ylabel("Persentase (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "03b_diversity_codon_stats.png", dpi=220)
    plt.close()
    logger.info("Selesai Langkah 3b.")


if __name__ == "__main__":
    main()
