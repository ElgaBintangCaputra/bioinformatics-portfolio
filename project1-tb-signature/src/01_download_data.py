"""
01_download_data.py
====================
Langkah 1: Mengunduh data ekspresi gen dari NCBI GEO (Gene Expression Omnibus).

Dataset yang dipakai
--------------------
DISCOVERY (data untuk "menemukan" gen & melatih model) — bagian dari SuperSeries
GSE19491 (Berry et al., Nature 2010). Ketiganya memakai platform yang sama
(GPL6947 = Illumina HumanHT-12 V3):
  * GSE19444 : kohort London "training set"  (TB aktif, TB laten, sehat)
  * GSE19439 : kohort London "test set"      (TB aktif, TB laten, sehat)
  * GSE19442 : kohort Afrika Selatan         (TB aktif, TB laten)

VALIDASI EKSTERNAL (data yang TIDAK PERNAH dilihat model saat latihan):
  * GSE37250 : Kaforou et al., PLoS Med 2013 — Malawi & Afrika Selatan,
               pasien dengan/tanpa HIV (TB aktif, TB laten, penyakit lain).
               Platform GPL10558 = Illumina HumanHT-12 V4 (platform BERBEDA).

File yang diunduh
-----------------
1. "Series matrix" (*.txt.gz): tabel ekspresi (baris = probe, kolom = sampel)
   ditambah metadata sampel (baris yang diawali "!Sample_").
2. Anotasi platform (*.annot.gz): tabel yang memetakan ID probe -> simbol gen.

Skrip ini sengaja hanya memakai pustaka bawaan Python (urllib), jadi mudah
dipahami. Jika file sudah ada, tidak diunduh ulang (hemat waktu & kuota).
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

# Folder tujuan: <project>/data/raw
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Alamat server GEO. Pola folder GEO: GSE19444 -> folder "GSE19nnn"
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

SERIES = ["GSE19444", "GSE19439", "GSE19442", "GSE37250"]
PLATFORMS = ["GPL6947", "GPL10558"]


def folder_geo(accession: str) -> str:
    """Ubah 'GSE19444' -> 'GSE19nnn' (3 digit terakhir diganti 'nnn')."""
    prefix = accession[:3]           # 'GSE' atau 'GPL'
    angka = accession[3:]            # '19444'
    if len(angka) <= 3:
        return f"{prefix}nnn"
    return f"{prefix}{angka[:-3]}nnn"


def unduh(url: str, tujuan: Path) -> None:
    """Unduh satu file dari URL ke path tujuan (lewati jika sudah ada)."""
    if tujuan.exists() and tujuan.stat().st_size > 0:
        print(f"  [sudah ada] {tujuan.name}")
        return
    print(f"  [unduh] {url}")
    tmp = tujuan.with_suffix(tujuan.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(tujuan)  # rename di akhir: file setengah jadi tidak dianggap selesai
    print(f"           -> {tujuan.stat().st_size / 1e6:.1f} MB")


def cek_label(path: Path, n_baris: int = 3) -> None:
    """Tampilkan beberapa baris metadata karakteristik sampel untuk verifikasi."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        ditampilkan = 0
        for line in f:
            if line.startswith("!Sample_characteristics_ch1") or line.startswith("!Sample_source_name_ch1"):
                isi = line.strip().split("\t")
                contoh = sorted(set(x.strip('"') for x in isi[1:]))[:6]
                print(f"     {isi[0]}: {contoh}")
                ditampilkan += 1
                if ditampilkan >= n_baris:
                    break
            if line.startswith("!series_matrix_table_begin"):
                break


def main():
    print("== Mengunduh series matrix ==")
    for gse in SERIES:
        url = f"{GEO_FTP}/series/{folder_geo(gse)}/{gse}/matrix/{gse}_series_matrix.txt.gz"
        tujuan = RAW_DIR / f"{gse}_series_matrix.txt.gz"
        unduh(url, tujuan)
        cek_label(tujuan)

    print("== Mengunduh anotasi platform ==")
    for gpl in PLATFORMS:
        url = f"{GEO_FTP}/platforms/{folder_geo(gpl)}/{gpl}/annot/{gpl}.annot.gz"
        unduh(url, RAW_DIR / f"{gpl}.annot.gz")

    print("Selesai. File tersimpan di:", RAW_DIR)


if __name__ == "__main__":
    main()
