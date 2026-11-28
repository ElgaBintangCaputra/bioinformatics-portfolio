"""
run_all.py
==========
Menjalankan seluruh pipeline Project 1 (TB-Signature) secara berurutan dari langkah 1 hingga 6.
Catatan: Langkah 1 (unduh data) dan Langkah 4 (API g:Profiler) membutuhkan internet
hanya pada saat dijalankan pertama kali; selanjutnya hasil tersimpan di cache lokal.
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"

SCRIPTS = [
    "01_download_data.py",
    "02_preprocess.py",
    "03_differential_expression.py",
    "04_enrichment.py",
    "05_machine_learning.py",
    "06_boolean_network.py",
    "07_lodo_benchmark_ode.py",
]


def main():
    for name in SCRIPTS:
        script_path = SRC_DIR / name
        print(f"\n{'=' * 65}\nMenjalankan {name}...\n{'=' * 65}")
        res = subprocess.run([sys.executable, str(script_path)], check=False)
        if res.returncode != 0:
            raise SystemExit(f"Gagal pada {name} (exit code {res.returncode})")
    print("\nSemua langkah selesai! Jalankan dashboard dengan:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
