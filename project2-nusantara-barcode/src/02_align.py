"""
02_align.py
===========
Langkah 2: Multiple Sequence Alignment (MSA) & Trimming.

Kenapa perlu alignment?
Sekuens COI dari berbagai peneliti sering mulai dan berakhir di posisi sedikit
berbeda (ada yang 650 bp, ada yang 1200 bp). Alignment dengan MAFFT menyusun ulang
kolom-kolom homolog agar setiap posisi kolom mewakili basa leluhur yang sama.

Setelah itu dilakukan *trimming*:
1. Membuang kolom tepi/gap yang hanya terisi < 80% sekuens (`min_column_occupancy`).
2. Membuang sekuens pendek yang tidak menutupi >= 85% panjang alignment hasil trimming
   (`min_seq_coverage`), supaya matriks jarak K2P & pohon filogenetik bersih dan bebas artefak.
"""

import subprocess
from pathlib import Path

import pandas as pd
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from common import PROJECT_DIR, get_paths, load_config, setup_logger


def run_mafft(in_fasta: Path, out_fasta: Path, mafft_bat: Path, threads: int, logger) -> None:
    """Menjalankan binary MAFFT Windows (tools/mafft/mafft-win/mafft.bat)."""
    if not mafft_bat.exists():
        raise FileNotFoundError(f"MAFFT tidak ditemukan di {mafft_bat}")
    cmd = [
        "cmd.exe", "/c", str(mafft_bat),
        "--auto", "--thread", str(threads),
        str(in_fasta),
    ]
    logger.info(f"Menjalankan MAFFT: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"MAFFT gagal (code {res.returncode}): {res.stderr[:500]}")
    out_fasta.write_text(res.stdout, encoding="utf-8")


def trim_alignment(aln_fasta: Path, out_fasta: Path, min_col_occ: float, min_seq_cov: float, logger) -> list[str]:
    """Memangkas kolom ber-gap dan menyaring sekuens dengan cakupan rendah."""
    aln = AlignIO.read(aln_fasta, "fasta")
    n_seq = len(aln)
    n_col = aln.get_alignment_length()
    logger.info(f"Alignment awal: {n_seq} sekuens x {n_col} kolom")

    # 1. Pilih kolom yang terisi basa (bukan '-') minimal min_col_occ
    kept_cols = []
    for c in range(n_col):
        col_str = aln[:, c]
        non_gap = sum(1 for ch in col_str if ch != "-")
        if non_gap / n_seq >= min_col_occ:
            kept_cols.append(c)

    if not kept_cols:
        raise RuntimeError("Tidak ada kolom tersisa setelah trimming!")

    # 2. Bangun sekuens hasil pangkas & cek cakupan tiap sekuens
    kept_records = []
    kept_accessions = []
    n_kept_cols = len(kept_cols)
    for rec in aln:
        trimmed_seq = "".join(rec.seq[c] for c in kept_cols)
        non_gap = sum(1 for ch in trimmed_seq if ch != "-")
        if non_gap / n_kept_cols >= min_seq_cov:
            kept_records.append(SeqRecord(Seq(trimmed_seq), id=rec.id, description=""))
            kept_accessions.append(rec.id.split("|")[0])

    SeqIO.write(kept_records, out_fasta, "fasta")
    logger.info(
        f"Setelah trimming: {len(kept_records)} sekuens x {n_kept_cols} kolom "
        f"(dibuang {n_seq - len(kept_records)} sekuens pendek)"
    )
    return kept_accessions


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    logger.info("== Langkah 2: Multiple Sequence Alignment (MAFFT) & Trimming ==")

    raw_fasta = paths["processed_dir"] / "coi_raw.fasta"
    aln_raw = paths["processed_dir"] / "coi_aligned_raw.fasta"
    aln_trim = paths["processed_dir"] / "coi_aligned_trimmed.fasta"

    mafft_bat = PROJECT_DIR / "tools" / "mafft" / "mafft-win" / "mafft.bat"
    run_mafft(raw_fasta, aln_raw, mafft_bat, cfg["alignment"]["mafft_threads"], logger)

    kept_acc = trim_alignment(
        aln_raw,
        aln_trim,
        cfg["alignment"]["min_column_occupancy"],
        cfg["alignment"]["min_seq_coverage"],
        logger,
    )

    meta = pd.read_csv(paths["tables_dir"] / "01_sequence_metadata.csv")
    meta_trim = meta[meta["accession"].isin(kept_acc)].copy()
    # Pastikan hanya mempertahankan spesies yang masih punya >= min_per_species setelah trimming
    counts = meta_trim["species"].value_counts()
    valid_sp = counts[counts >= cfg["filters"]["min_per_species"]].index
    if len(valid_sp) < meta_trim["species"].nunique():
        dropped = set(meta_trim["species"].unique()) - set(valid_sp)
        logger.warning(f"Spesies dibuang pasca-trimming (< {cfg['filters']['min_per_species']} seq): {dropped}")
        meta_trim = meta_trim[meta_trim["species"].isin(valid_sp)].copy()
        valid_ids = set(meta_trim["accession"])
        recs = [r for r in SeqIO.parse(aln_trim, "fasta") if r.id.split("|")[0] in valid_ids]
        SeqIO.write(recs, aln_trim, "fasta")

    meta_trim.to_csv(paths["tables_dir"] / "02_aligned_metadata.csv", index=False)
    logger.info(f"Selesai Langkah 2: {len(meta_trim)} sekuens dari {meta_trim['species'].nunique()} spesies siap dianalisis.")


if __name__ == "__main__":
    main()
