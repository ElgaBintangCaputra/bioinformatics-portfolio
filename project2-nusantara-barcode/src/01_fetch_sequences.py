"""
01_fetch_sequences.py
=====================
Langkah 1: Mengunduh sekuens DNA barcode COI ikan konsumsi Indonesia dari NCBI GenBank.

Fitur penting:
1. Membaca daftar takson dan parameter dari config.yaml (prinsip Bio-HPC).
2. Mengunduh format GenBank lengkap (.gb) dan menyimpannya di data/raw/ sebagai cache
   agar analisis lanjutan bisa dijalankan 100% offline dan reprodusibel.
3. Mengekstrak metadata geografi: negara asal (geo_loc_name / country) dan koordinat
   lintang-bujur (lat_lon) jika tersedia di catatan GenBank.
4. Menyaring kualitas:
   - Panjang sekuens >= min_length (550 bp)
   - Fraksi basa ambigu (N, R, Y, ...) <= 1%
   - Memprioritaskan sampel dari Indonesia / Segitiga Terumbu Karang
   - Menghapus duplikat sekuens per spesies & membatasi max_per_species (8)
   - Membuang spesies yang punya < min_per_species (3) sekuens berkualitas.
"""

import re
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

from common import get_paths, load_config, setup_logger


def parse_lat_lon(text: str) -> tuple[float | None, float | None]:
    """Mengubah string GenBank seperti '8.75 S 115.17 E' menjadi (lat, lon) desimal."""
    if not text:
        return None, None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([NS])\s+([0-9]+(?:\.[0-9]+)?)\s*([EW])", text, re.I)
    if not m:
        return None, None
    lat = float(m.group(1)) * (-1 if m.group(2).upper() == "S" else 1)
    lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
    return lat, lon


def clean_country(raw_loc: str) -> str:
    """Ambil nama negara dari qualifier geo_loc_name/country (mis. 'Indonesia: Bali' -> 'Indonesia')."""
    if not raw_loc:
        return "Unknown"
    return raw_loc.split(":")[0].strip()


def fetch_species_genbank(species: str, cfg: dict, raw_dir: Path, logger) -> list[SeqRecord]:
    """Unduh catatan GenBank untuk satu spesies (atau baca dari cache lokal jika sudah ada)."""
    safe_name = species.replace(" ", "_")
    cache_gb = raw_dir / f"{safe_name}.gb"

    if cache_gb.exists() and cache_gb.stat().st_size > 0:
        logger.info(f"  [cache] {species}")
        return list(SeqIO.parse(cache_gb, "genbank"))

    gene_or = " OR ".join(cfg["gene"]["query_terms"])
    min_r = cfg["gene"]["min_record_length"]
    max_r = cfg["gene"]["max_record_length"]
    query = f'"{species}"[Organism] AND ({gene_or}) AND {min_r}:{max_r}[Sequence Length]'

    handle = Entrez.esearch(db="nucleotide", term=query, retmax=cfg["entrez"]["retmax_search"])
    res = Entrez.read(handle)
    handle.close()
    ids = res.get("IdList", [])
    time.sleep(cfg["entrez"]["delay_seconds"])

    if not ids:
        logger.warning(f"  [kosong] Tidak ada hit GenBank untuk {species}")
        return []

    logger.info(f"  [unduh] {species}: {len(ids)} catatan ditemukan...")
    fetch_h = Entrez.efetch(db="nucleotide", id=",".join(ids), rettype="gb", retmode="text")
    gb_text = fetch_h.read()
    fetch_h.close()
    time.sleep(cfg["entrez"]["delay_seconds"])

    cache_gb.write_text(gb_text, encoding="utf-8")
    return list(SeqIO.parse(cache_gb, "genbank"))


def extract_and_filter(records: list[SeqRecord], species: str, family: str, cfg: dict) -> list[dict]:
    """Saring kualitas sekuens & ekstrak metadata per catatan GenBank."""
    min_len = cfg["filters"]["min_length"]
    max_amb = cfg["filters"]["max_ambiguous_frac"]
    prefer_country = cfg["filters"]["prefer_country"].lower()

    candidates = []
    seen_seqs = set()
    for rec in records:
        seq_str = str(rec.seq).upper().replace("-", "")
        if len(seq_str) < min_len:
            continue
        amb_count = sum(1 for c in seq_str if c not in "ACGT")
        amb_frac = amb_count / len(seq_str)
        if amb_frac > max_amb:
            continue

        # Ekstrak qualifier dari feature 'source'
        country_raw = ""
        lat_lon_raw = ""
        voucher = ""
        for feat in rec.features:
            if feat.type == "source":
                q = feat.qualifiers
                country_raw = (q.get("geo_loc_name") or q.get("country") or [""])[0]
                lat_lon_raw = (q.get("lat_lon") or [""])[0]
                voucher = (q.get("specimen_voucher") or q.get("isolate") or [""])[0]
                break

        country = clean_country(country_raw)
        lat, lon = parse_lat_lon(lat_lon_raw)
        # Hindari duplikat urutan basa persis kecuali punya lokasi berbeda yang berguna
        if seq_str in seen_seqs and country == "Unknown":
            continue
        seen_seqs.add(seq_str)

        # Skor prioritas: sampel Indonesia & punya koordinat didahulukan
        prio = 0
        if prefer_country in country.lower():
            prio += 10
        elif country != "Unknown":
            prio += 3
        if lat is not None:
            prio += 2

        candidates.append({
            "accession": rec.id,
            "species": species,
            "family": family,
            "genus": species.split()[0],
            "length": len(seq_str),
            "ambiguous_frac": round(amb_frac, 4),
            "country": country,
            "country_detail": country_raw,
            "lat": lat,
            "lon": lon,
            "voucher": voucher,
            "priority": prio,
            "sequence": seq_str,
        })

    # Urutkan berdasarkan prioritas lalu panjang sekuens, ambil maksimal max_per_species
    candidates.sort(key=lambda r: (r["priority"], r["length"]), reverse=True)
    return candidates[: cfg["filters"]["max_per_species"]]


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    Entrez.email = cfg["entrez"]["email"]

    logger.info("== Langkah 1: Mengunduh & Menyaring Sekuens COI dari NCBI GenBank ==")
    all_rows = []
    min_sp = cfg["filters"]["min_per_species"]

    for family, species_list in cfg["taxa"].items():
        for sp in species_list:
            recs = fetch_species_genbank(sp, cfg, paths["raw_dir"], logger)
            filtered = extract_and_filter(recs, sp, family, cfg)
            if len(filtered) < min_sp:
                logger.warning(f"  [drop] {sp}: hanya {len(filtered)} sekuens lolos (< {min_sp})")
                continue
            all_rows.extend(filtered)

    df = pd.DataFrame(all_rows)
    # Tulis FASTA dengan header informatif: Accession|Species|Family
    fasta_path = paths["processed_dir"] / "coi_raw.fasta"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            sp_tag = r["species"].replace(" ", "_")
            seq_id = f"{r['accession']}|{sp_tag}|{r['family']}"
            f.write(f">{seq_id}\n{r['sequence']}\n")

    meta_df = df.drop(columns=["sequence", "priority"])
    meta_df.to_csv(paths["tables_dir"] / "01_sequence_metadata.csv", index=False)

    n_sp = meta_df["species"].nunique()
    n_id = (meta_df["country"].str.lower() == "indonesia").sum()
    logger.info(f"Selesai Langkah 1: {len(meta_df)} sekuens dari {n_sp} spesies ({n_id} berlabel Indonesia).")


if __name__ == "__main__":
    main()
