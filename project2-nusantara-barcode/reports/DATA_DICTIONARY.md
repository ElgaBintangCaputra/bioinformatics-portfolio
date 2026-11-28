# 📋 Kamus Data & Dokumentasi Input–Proses–Output (Project 2: NusantaraBarcode)

Dokumen ini merinci **setiap file input, proses transformasi, dan file output** pada pipeline **NusantaraBarcode**.

---

## 1. Alur Lengkap Input → Proses → Output

| Tahap | Skrip | Input Utama | Proses yang Dilakukan | Output yang Dihasilkan |
|---|---|---|---|---|
| **1. Akuisisi & QC Sekuens** | `src/01_fetch_sequences.py` | `config.yaml` (29 spesies ikan) + NCBI Entrez | Unduh catatan GenBank (`.gb`) ber-cache; ekstrak `country` & `lat_lon`; saring panjang (`>= 550 bp`), basa ambigu (`<= 1%`), hapus duplikat tanpa lokasi, prioritaskan sampel Indonesia (`max 8/spesies`) | `data/raw/*.gb`, `data/processed/coi_raw.fasta`, `results/tables/01_sequence_metadata.csv` |
| **2. Alignment & Trimming** | `src/02_align.py` | `data/processed/coi_raw.fasta` | Multiple Sequence Alignment dengan `MAFFT --auto`; buang kolom dengan cakupan `< 80%` dan sekuens dengan cakupan `< 85%` | `data/processed/coi_aligned_raw.fasta`, `coi_aligned_trimmed.fasta` (229 × 636 bp), `results/tables/02_aligned_metadata.csv` |
| **3. Jarak K2P & Barcode Gap** | `src/03_distance_barcode_gap.py` | `coi_aligned_trimmed.fasta` | Hitung jarak evolusi Kimura 2-Parameter (transisi vs transversi) berpasangan; evaluasi intra- vs interspesifik dan *nearest neighbor* per spesies | `results/tables/03_k2p_distance_matrix.csv`, `03_barcode_gap_per_species.csv`, `03_distance_summary.csv`, `results/figures/03_barcode_gap_plot.png` |
| **4. Rekonstruksi Filogeni** | `src/04_phylogeny.py` | `coi_aligned_trimmed.fasta`, `03_k2p_distance_matrix.csv` | Bangun pohon Neighbor-Joining (Biopython) dan Maximum-Likelihood **IQ-TREE 2** (`-m MFP -B 1000`), *rooting* pada outgroup `Scombridae` | `results/trees/nj_full.nwk`, `iqtree_species.treefile`, `iqtree_full.treefile`, `results/figures/04_*.png` & `.pdf` |
| **5. Genetika Populasi & Audit** | `src/03b_diversity_selection.py` | `coi_aligned_trimmed.fasta`, `03_k2p_distance_matrix.csv`, `iqtree_full.treefile` | Hitung jumlah haplotipe ($h$), $H_d$, $\pi$, $S$, *Tajima's D*; statistik posisi kodon 1/2/3 & *sliding window*; deteksi otomatis *misannotation* GenBank; uji monofili pohon ML | `results/tables/03b_population_genetics.csv`, `03b_codon_position_stats.csv`, `03b_genbank_audit.csv`, `03b_monophyly_check.csv`, `results/figures/03b_diversity_codon_stats.png` |
| **6. Klasifikasi ML 6-mer** | `src/05_ml_species_id.py` | `coi_raw.fasta` (sekuens lolos QC) | Ekstraksi vektor frekuensi 6-mer (4.096 dimensi); 3-Fold Stratified CV membandingkan Random Forest, 1-NN Cosine, dan Linear SVM | `results/tables/05_ml_comparison.csv`, `results/models/species_id_model.joblib`, `results/figures/05_confusion_matrix.png` |
| **7. Peta Geografis** | `src/06_map.py` | `02_aligned_metadata.csv` | Gabungkan koordinat asli GenBank & *centroid* negara secara transparan untuk visualisasi sebaran Indo-Pasifik | `results/tables/06_metadata_with_coords.csv`, `results/figures/06_sampling_map.html`, `06_country_distribution.png` |

---

## 2. Kamus Kolom Output Penting

- **`03_barcode_gap_per_species.csv`**: `species`, `family`, `n_seq`, `mean_intra_k2p`, `max_intra_k2p`, `min_inter_k2p`, `nearest_neighbor`, `barcode_gap` (`min_inter - max_intra`), `has_gap` (`True`/`False`).
- **`03b_population_genetics.csv`**: `h` (jumlah haplotipe unik), `Hd` (keragaman haplotipe 0–1), `S` (situs segregasi), `pi` (keragaman nukleotida per situs), `tajimas_D` (statistik uji netralitas).
- **`03b_codon_position_stats.csv`**: `Posisi_Kodon` (1, 2, 3), `Jumlah_Situs` (212 kodon), `Situs_Polimorfik`, `Persen_Polimorfik`, `GC_Persen`, `ORF_Frame_Terpilih`, `Total_Internal_Stop_ORF` (verifikasi translasi Kode Genetik Mitokondria Vertebrata Tabel 2 NCBI = 0 stop codon internal).
- **`03b_genbank_audit.csv`**: `seq_id`, `accession`, `nominal_species`, `nominal_family`, `mean_dist_to_own_species`, `nearest_other_seq`, `nearest_other_species`, `min_dist_to_other`, `status_audit`, `action_recommended` (`Exclude_From_Reference` / `Require_Nuclear_Marker` / `Review_Taxonomy`).
- **`05_ml_ablation_audit.csv`**: Studi ablasi 3-Fold Stratified CV (`1-NN Cosine`, `Linear SVM`, `Random Forest`) pada 3 rezim dataset: (1) Raw GenBank ($n=229$), (2) Post-Audit Clean ($n=227$, tanpa spesimen mislabel lintas-famili), dan (3) Post-Audit + Resolusi Kompleks Introgresi *Thunnus* ($n=227$).
