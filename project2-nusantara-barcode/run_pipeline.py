"""
run_pipeline.py
===============
ORKESTRATOR PIPELINE NusantaraBarcode (berbasis config.yaml).
Menjalankan langkah 1 -> 6 secara berurutan dengan logging dan fitur lewati-jika-sudah-ada
(mirip cara kerja Snakemake/Nextflow di lingkungan Bio-HPC).

Penggunaan:
  python run_pipeline.py          # jalankan langkah yang outputnya belum ada
  python run_pipeline.py --force  # jalankan ulang semua langkah dari awal
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.common import PROJECT_DIR, get_paths, load_config, setup_logger

STEPS = [
    ("01_fetch_sequences.py", "tables", "01_sequence_metadata.csv"),
    ("02_align.py", "processed", "coi_aligned_trimmed.fasta"),
    ("03_distance_barcode_gap.py", "tables", "03_barcode_gap_per_species.csv"),
    ("04_phylogeny.py", "figures", "04_species_tree_ml.png"),
    ("03b_diversity_selection.py", "tables", "03b_genbank_audit.csv"),
    ("05_ml_species_id.py", "tables", "05_ml_comparison.csv"),
    ("06_map.py", "figures", "06_sampling_map.html"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Jalankan ulang semua langkah.")
    args = parser.parse_args()

    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])

    dir_lookup = {
        "tables": paths["tables_dir"],
        "processed": paths["processed_dir"],
        "figures": paths["figures_dir"],
    }

    for script, out_kind, out_file in STEPS:
        target = dir_lookup[out_kind] / out_file
        if target.exists() and not args.force:
            logger.info(f"[skip] {script} -> {out_file} sudah ada.")
            continue
        logger.info(f"[run] Menjalankan {script}...")
        res = subprocess.run([sys.executable, str(PROJECT_DIR / "src" / script)], check=False)
        if res.returncode != 0:
            raise SystemExit(f"Pipeline berhenti pada {script} (exit code {res.returncode})")

    logger.info("Pipeline NusantaraBarcode selesai seluruhnya! Jalankan `streamlit run app.py`.")


if __name__ == "__main__":
    main()
