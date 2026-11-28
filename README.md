# 🧬 Computational Biology and Bioinformatics — Research Portfolio

Kumpulan **4 Proyek Riset Terintegrasi** (*end-to-end*, berbasis data publik & kode Python reprodusibel) yang mencakup transkriptomik klinis, bioinformatika molekuler & filogenetika, *network pharmacology* & *cheminformatics*, serta pemodelan stokastik konservasi kandidat publikasi.

---

## 📁 Peta Proyek & Dokumentasi Input–Proses–Output

| Proyek | Fokus Bidang Riset | Kamus Data (Input–Proses–Output) | Dokumentasi & Dashboard |
|---|---|---|---|
| **[`project1-tb-signature/`](project1-tb-signature/README.md)** | **Transkriptomik, Nested CV Anti-Leakage, & Systems Biology** (147 Discovery → 537 Validasi Eksternal `GSE37250`, Nested 5×4-Fold CV, Strict LODO, Single-Sample vs Cohort Calibration, Benchmark *Sweeney 3-Gene* Utuh, Boolean Network & ODE) | [`DATA_DICTIONARY.md`](project1-tb-signature/reports/DATA_DICTIONARY.md) | [`README.md`](project1-tb-signature/README.md) • [`app.py`](project1-tb-signature/app.py) |
| **[`project2-nusantara-barcode/`](project2-nusantara-barcode/README.md)** | **Bioinformatika Molekuler, Filogenetika, Genetika Populasi, & Workflow Automation** (229 sekuens COI, MAFFT, IQ-TREE 2, K2P, Verifikasi ORF Mitokondria Tabel 2, Audit Presisi Asimetris GenBank, Ablasi 6-mer ML `99.56%`, `Snakefile`) | [`DATA_DICTIONARY.md`](project2-nusantara-barcode/reports/DATA_DICTIONARY.md) | [`README.md`](project2-nusantara-barcode/README.md) • [`app.py`](project2-nusantara-barcode/app.py) |
| **[`project3-jamunet/`](project3-jamunet/README.md)** | **Cheminformatics, Network Pharmacology, QSAR ML + Applicability Domain, & Ensemble 3D** (27 senyawa jamu, RDKit, ChEMBL, Open Targets, STRING PPI, MedChem Lead Score, QSAR DPP4 Scaffold-Split & GroupCV5, PAINS vs Brenk, Ensemble 10 Konformer 3D `.sdf`) | [`DATA_DICTIONARY.md`](project3-jamunet/reports/DATA_DICTIONARY.md) | [`README.md`](project3-jamunet/README.md) • [`app.py`](project3-jamunet/app.py) |
| **[`NusantaraRhino/`](NusantaraRhino/README.md)** | **Riset Kandidat Publikasi — Integrated Stochastic Population & Genetic Modeling (*Rhinoceros sondaicus*)** (IBM 3-Mesin, 20 lokus IBD, Kopula Bencana $\rho$, 5 Skenario Faktorial A–D, Latin Hypercube Sampling & True PRCC, 95% Wilson CI, Vortex 10 Spec Bridge) | [`DATA_DICTIONARY.md`](NusantaraRhino/reports/DATA_DICTIONARY.md) • [`RESEARCH_PROTOCOL_v1.0.md`](NusantaraRhino/docs/RESEARCH_PROTOCOL_v1.0.md) | [`MANUSCRIPT_DRAFT.md`](NusantaraRhino/manuscript/MANUSCRIPT_DRAFT.md) • [`app.py`](NusantaraRhino/app.py) |

---

## 🚀 Cara Menjalankan & Memverifikasi

```powershell
# 1. Buat virtual environment & pasang dependensi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Jalankan seluruh unit test komprehensif (21/21 passed)
pytest project1-tb-signature/tests -v
pytest project2-nusantara-barcode/tests -v
pytest project3-jamunet/tests -v
pytest NusantaraRhino/tests -v

# 3. Jalankan dashboard interaktif Streamlit di salah satu proyek
streamlit run project1-tb-signature/app.py
```

---

## ⚠️ Catatan Reprodusibilitas & Data Besar
File mentah berukuran besar (mis. series matrix NCBI GEO hingga >100 MB), binary eksternal (`MAFFT`/`IQ-TREE`), cache API, dan model `.joblib` tidak disertakan di repositori ini via `.gitignore`. Seluruh data mentah dapat diunduh ulang otomatis melalui skrip `01_*.py` masing-masing proyek, sementara tabel dan grafik hasil analisis sudah tersedia di tiap folder `results/` dengan dukungan *fallback* pada dashboard `app.py` dan unit test sehingga tetap aman dijalankan langsung setelah `git clone`.
