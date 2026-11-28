"""
run_all.py — Menjalankan verifikasi unit test, eksperimen simulasi, dan pembuatan Figure 1–6.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    steps = [
        [sys.executable, "-m", "pytest", "tests", "-v"],
        [sys.executable, "src/experiments/run_experiments.py"],
        [sys.executable, "src/visualization/make_figures.py"],
    ]
    for cmd in steps:
        print("\n==>", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=ROOT, check=False)
        if res.returncode != 0:
            raise SystemExit(f"Gagal pada perintah: {' '.join(cmd)}")
    print("\nNusantaraRhino selesai dijalankan! Buka dashboard dengan: streamlit run app.py")


if __name__ == "__main__":
    main()
