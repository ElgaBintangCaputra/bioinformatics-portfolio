"""
run_all.py
==========
Menjalankan seluruh pipeline Project 3 (JamuNet) dari langkah 1 hingga 7.
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"

SCRIPTS = [
    "01_compounds.py",
    "02_druglikeness.py",
    "03_targets.py",
    "04_disease_genes.py",
    "05_network.py",
    "06_enrichment.py",
    "07_similarity_ml.py",
    "08_qsar_3d_conformers.py",
]


def main():
    for name in SCRIPTS:
        print(f"\n{'=' * 65}\nMenjalankan {name}...\n{'=' * 65}")
        res = subprocess.run([sys.executable, str(SRC_DIR / name)], check=False)
        if res.returncode != 0:
            raise SystemExit(f"Gagal pada {name} (exit code {res.returncode})")
    print("\nSemua langkah JamuNet selesai! Jalankan dashboard dengan:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
