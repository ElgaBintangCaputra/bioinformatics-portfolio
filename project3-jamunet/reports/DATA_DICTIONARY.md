# 📋 Kamus Data & Dokumentasi Input–Proses–Output (Project 3: JamuNet)

Dokumen ini merinci **setiap file input, proses transformasi, dan file output** pada pipeline **JamuNet**.

---

## 1. Alur Lengkap Input → Proses → Output

| Tahap | Skrip | Input Utama | Proses yang Dilakukan | Output yang Dihasilkan |
|---|---|---|---|---|
| **1. Akuisisi Senyawa** | `src/01_compounds.py` | `data/plant_compounds_curated.csv`, `data/antidiabetic_drugs.csv` | Tarik struktur kanonik (SMILES, rumus molekul, InChIKey, CID) dari PubChem PUG-REST dengan cache lokal | `data/cache/pubchem/*`, `results/tables/01_compounds_pubchem.csv`, `01_plant_compound_table.csv`, `01_antidiabetic_drugs_pubchem.csv` |
| **2. Drug-likeness & 2D** | `src/02_druglikeness.py` | `01_compounds_pubchem.csv` | Hitung deskriptor RDKit (MW, cLogP, TPSA, HBD, HBA, RotBonds, QED), evaluasi Lipinski & Veber, gambar grid 2D | `results/tables/02_druglikeness.csv`, `results/figures/02_compound_structures_grid.png`, `02_druglikeness_overview.png` |
| **3. Target Manusia** | `src/03_targets.py` | `01_compounds_pubchem.csv` | Cocokkan InChIKey ke ChEMBL; tarik bioaktivitas *Homo sapiens* (`pChEMBL >= 5.5`) & petakan target ke simbol gen HGNC secara batch | `data/cache/chembl/*`, `results/tables/03_compound_targets.csv`, `results/figures/03_targets_per_compound.png` |
| **4. Gen Penyakit T2D** | `src/04_disease_genes.py` | `03_compound_targets.csv` + Open Targets GraphQL | Tarik 3.000 gen Diabetes Tipe 2 (`MONDO_0005148`, `score >= 0.02`), hitung irisan Target Jamu ∩ T2D (`47` gen), gambar diagram Venn | `data/cache/opentargets/*`, `results/tables/04_t2d_genes.csv`, `04_overlapping_targets.csv`, `results/figures/04_target_t2d_overlap.png` |
| **5. Jaringan & Hub PPI** | `src/05_network.py` | `04_overlapping_targets.csv`, `03_compound_targets.csv` | Bangun jaringan tripartit (Tanaman→Senyawa→Target→T2D) interaktif PyVis; tarik PPI STRING (`score >= 0.4`), hitung *degree*, *betweenness*, *closeness*, & komunitas | `data/cache/string/*`, `results/tables/05_string_ppi_edges.csv`, `05_hub_genes.csv`, `results/figures/05_jamunet_interactive.html`, `05_network_overview.png` |
| **6. Pengayaan Jalur** | `src/06_enrichment.py` | `04_overlapping_targets.csv` | Kueri g:Profiler (`GO:BP`, `KEGG`, `REAC`) pada 47 gen irisan | `data/cache/gprofiler/*`, `results/tables/06_pathway_enrichment.csv`, `results/figures/06_pathway_enrichment.png` |
| **7. Ruang Kimia & Ranking** | `src/07_similarity_ml.py` | Senyawa jamu + 9 obat diabetes | Morgan Fingerprint (2048-bit), kemiripan Tanimoto vs obat resmi, PCA ruang kimia, & skor gabungan transparan (`0.45*Net + 0.30*QED + 0.25*Tanimoto`) | `results/tables/07_candidate_ranking.csv`, `07_chemical_space_pca.csv`, `results/figures/07_chemical_space_pca.png` |
| **8. QSAR ML, PAINS & 3D** | `src/08_qsar_3d_conformers.py` | `01_compounds_pubchem.csv`, `07_candidate_ranking.csv` | Filter medisinal PAINS & Brenk; unduh 392 inhibitor DPP4 (`CHEMBL284`), latih Random Forest QSAR dengan **Bemis-Murcko Scaffold Split**; bangkitkan konformer 3D `ETKDGv3` + minimisasi `MMFF94` untuk Top 5 | `data/cache/chembl_qsar/*`, `results/tables/08_pains_brenk_alerts.csv`, `08_qsar_metrics.csv`, `08_jamu_qsar_predictions.csv`, `08_3d_conformers.csv`, `results/structures/*.sdf`, `results/figures/08_qsar_pains_summary.png` |

---

## 2. Kamus Kolom Output Penting

- **`02_druglikeness.csv`**: `mw`, `clogp`, `tpsa`, `hbd`, `hba`, `rot_bonds`, `qed` (0–1), `lipinski_violations`, `lipinski_pass`, `veber_pass`.
- **`05_hub_genes.csv`**: `gene_symbol`, `degree`, `degree_centrality`, `betweenness_centrality`, `closeness_centrality`, `community`, `n_compounds`, `ot_score`.
- **`07_candidate_ranking.csv`**: `n_t2d_targets`, `qed`, `closest_drug`, `max_tanimoto`, `n_pains_alerts`, `n_brenk_alerts`, `medchem_status`, `combined_score` (skor komposit klasik), `medchem_lead_score` (skor prioritas lead terkalibrasi dengan peredaman log-target & penalti PAINS/Brenk), `candidate_category`.
- **`08_qsar_metrics.csv`**: `CV5_ROC_AUC`, `GroupCV5_Scaffold_ROC_AUC`, `ScaffoldSplit_ROC_AUC`, `ScaffoldSplit_R2`, `ScaffoldSplit_RMSE`, `ScaffoldSplit_Pearson_r`.
- **`08_jamu_qsar_predictions.csv`**: `pred_dpp4_pchembl`, `pred_dpp4_prob_active`, `max_tanimoto_dpp4_train`, `applicability_domain` (`Inside AD (>=0.25)` vs `Extrapolation / Natural Product (<0.25)`), `n_pains_alerts`, `pains_detail`, `n_brenk_alerts`, `brenk_detail`.
- **`08_3d_conformers.csv`**: `n_atoms_with_H`, `n_conformers_sampled` (10 konformer `ETKDGv3`), `mmff94_energy_kcal_mol` (energi minimum), `conformer_energy_span_kcal_mol` (selisih energi maks-min antar-konformer), `sdf_file`.
