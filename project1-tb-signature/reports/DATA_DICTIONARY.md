# 📋 Kamus Data & Dokumentasi Input–Proses–Output (Project 1: TB-Signature)

Dokumen ini menjelaskan secara lengkap **setiap file input, proses transformasi, dan file output** pada pipeline **TB-Signature**.

---

## 1. Alur Lengkap Input → Proses → Output

| Tahap | Skrip | Input Utama | Proses yang Dilakukan | Output yang Dihasilkan |
|---|---|---|---|---|
| **1. Akuisisi Data** | `src/01_download_data.py` | URL FTP NCBI GEO (`GSE19444`, `GSE19439`, `GSE19442`, `GSE37250`, `GPL6947`, `GPL10558`) | Unduh aman (`.part` → rename) ke cache lokal; cek label karakteristik sampel | `data/raw/*_series_matrix.txt.gz`, `data/raw/GPL*.annot.gz` |
| **2. Preprocessing & QC** | `src/02_preprocess.py` | File `.txt.gz` & `.annot.gz` di `data/raw/` | Ekstrak label klinis dari `Sample_source_name_ch1`, transformasi `log2`, *Quantile Normalization* per kohort, mapping probe → simbol gen (ambil probe ekspresi rata-rata tertinggi), irisan gen antar platform (`18.919`), filter top 50% varians di Discovery (`9.459` gen), PCA QC | `data/processed/discovery_expr.parquet`, `discovery_meta.csv`, `validation_expr.parquet`, `validation_meta.csv`, `results/tables/02_sample_summary.csv`, `02_pca_discovery.csv`, `results/figures/02_qc_pca.png` |
| **3. Ekspresi Diferensial** | `src/03_differential_expression.py` | `discovery_expr.parquet`, `discovery_meta.csv` | Hitung `log2FC`, uji Welch's t-test (`equal_var=False`), koreksi berganda Benjamini–Hochberg (`fdr_bh`), klasifikasi Up/Down (`FDR < 0.05`, `|log2FC| >= 0.5`) | `results/tables/03_differential_expression.csv`, `03_top30_degs.csv`, `results/figures/03_volcano_plot.png`, `03_heatmap_top30.png` |
| **4. Pengayaan Jalur** | `src/04_enrichment.py` | `03_differential_expression.csv` (535 gen Up vs 9.459 *background*) | Kueri g:Profiler REST API (`GO:BP`, `KEGG`, `REAC`), simpan respons ke cache lokal | `data/cache/enrichment_gprofiler.json`, `results/tables/04_pathway_enrichment.csv`, `results/figures/04_pathway_enrichment.png` |
| **5. Machine Learning & Validasi Eksternal** | `src/05_machine_learning.py` | `discovery_*` & `validation_*` | Normalisasi z-score per sampel + kalibrasi baseline gen *unsupervised* per kohort; 5-Fold Stratified CV dengan `StandardScaler` + `SelectKBest` di dalam `Pipeline` (anti-*leakage*); seleksi **8-Gen Signature** dari koefisien L1; *Permutation Importance*; uji buta ke `GSE37250` (`n=537`) & subgrup HIV-/HIV+ | `results/tables/05_cv_comparison.csv`, `05_signature_genes.csv`, `05_external_validation.csv`, `05_validation_predictions.csv`, `results/models/tb_signature_model.joblib`, `results/figures/05_roc_curves.png`, `05_signature_importance.png` |
| **6. Boolean Network** | `src/06_boolean_network.py` | Definisi aturan logika 8-node imun interferon (`Mtb_Load`..`SOCS1`) | Simulasi sinkron seluruh $2^8 = 256$ keadaan awal untuk menemukan seluruh *attractor* & lintasan infeksi | `results/tables/06_boolean_attractors.csv`, `06_boolean_trajectory_infection.csv`, `results/figures/06_boolean_network.png` |
| **7. LODO, Benchmark, & ODE** | `src/07_lodo_benchmark_ode.py` | `discovery_*`, `validation_*`, `tb_signature_model.joblib` | Uji *Leave-One-Dataset-Out* antar 3 kohort Discovery, *benchmark* vs skor literatur *Sweeney*, simulasi model kontinu ODE (`solve_ivp`) 3 rezim klinis | `results/tables/07_lodo_validation.csv`, `07_benchmark_sweeney3.csv`, `07_ode_simulation.csv`, `results/figures/07_benchmark_and_ode.png` |

---

## 2. Kamus Kolom Output Penting

- **`03_differential_expression.csv`**: `gene` (simbol gen HGNC), `mean_active_tb` & `mean_control` (rata-rata ekspresi log2 ternormalisasi kuantil), `log2FC` (selisih rata-rata), `t_stat` (statistik Welch), `p_value`, `fdr` (q-value Benjamini–Hochberg), `status` (`Naik di TB (Up)` / `Turun di TB (Down)` / `Tidak Signifikan`).
- **`05_cv_comparison.csv`**: Perbandingan 5-Fold Stratified CV (`RandomForest`, `LogisticRegression (L1)`, `Linear SVM`) dan `8-Gen Signature (Nested 5x4-Fold CV)` yang 100% bebas *post-selection feature leakage*.
- **`05_signature_genes.csv`**: `gene`, `coef_logreg`, `intercept_logreg`, `scaler_mean`, `scaler_scale`, `frozen_disc_raw_mean`, `frozen_disc_raw_std`, `perm_importance_auc` (Discovery), `perm_importance_ext_auc` (*Out-of-Sample* di `GSE37250`), `log2FC_discovery`, `fdr_discovery`.
- **`05_external_validation.csv`**: `Evaluasi` (kelompok uji di `GSE37250`), `N_Sampel`, `N_Active_TB`, `AUC` (*Unsupervised Target-Cohort Calibration*), `AUC_SingleSample_Frozen` (*True Single-Sample Frozen-Discovery Calibration*), `Accuracy`, `Sensitivity`, `Specificity`, `F1`.
- **`07_lodo_validation.csv`**: `Kohort_Uji` (dataset yang disisihkan sebagai uji murni), `ROC_AUC_Strict_LODO` & `PR_AUC_Strict_LODO` (seleksi 8 gen diulang murni dari 2 kohort latih), `ROC_AUC` & `PR_AUC` (panel konsensus 8 gen), `Strict_Fold_Genes`.
- **`07_benchmark_sweeney3.csv`**: `Tugas_Evaluasi`, `Sweeney_Gen_Tersedia` (`GBP5+DUSP3+KLF2`), `AUC_TB_Signature_8Gen`, `AUC_Sweeney_3Gen` (skor 3-gen utuh), `AUC_GBP5_Single` (skor 1-gen `GBP5`), beserta nilai `PR_AUC` masing-masing.
- **`07_ode_simulation.csv`**: `Skenario`, `Hari` (0–40), `Beban_Bakteri_B`, `Interferon_I`, `Efektor_Makrofag_M`, `Rem_SOCS1_S`.
