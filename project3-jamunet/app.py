"""
app.py
======
Dashboard interaktif Streamlit untuk Project 3: JamuNet.
Menampilkan:
  1. Eksplorasi Tanaman Jamu & Struktur Kimia 2D
  2. Profil Drug-likeness (Lipinski / Veber / QED)
  3. Irisan Target ChEMBL & Gen Diabetes Tipe 2 (Open Targets)
  4. Jaringan Interaktif Tanaman-Senyawa-Target-T2D (pyvis) & Hub Genes STRING
  5. Pengayaan Jalur Biologis (g:Profiler)
  6. Ruang Kimia & Peringkat Akhir Kandidat Jamu
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"

st.set_page_config(
    page_title="JamuNet | Network Pharmacology Jamu & Diabetes Tipe 2",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 JamuNet: Network Pharmacology Tanaman Jamu untuk Diabetes Tipe 2")
st.caption(
    "Biologi Komputasi & In Silico Drug Discovery (PubChem + RDKit + ChEMBL + Open Targets + STRING + g:Profiler) "
    "• Untuk riset & edukasi (bukan saran medis)."
)


@st.cache_data
def load_tables():
    plant_tab = pd.read_csv(TABLE_DIR / "01_plant_compound_table.csv")
    dl = pd.read_csv(TABLE_DIR / "02_druglikeness.csv")
    ct = pd.read_csv(TABLE_DIR / "03_compound_targets.csv")
    overlap = pd.read_csv(TABLE_DIR / "04_overlapping_targets.csv")
    hubs = pd.read_csv(TABLE_DIR / "05_hub_genes.csv")
    enrich = pd.read_csv(TABLE_DIR / "06_pathway_enrichment.csv")
    rank = pd.read_csv(TABLE_DIR / "07_candidate_ranking.csv")
    pca = pd.read_csv(TABLE_DIR / "07_chemical_space_pca.csv")
    qsar_m = pd.read_csv(TABLE_DIR / "08_qsar_metrics.csv")
    qsar_p = pd.read_csv(TABLE_DIR / "08_jamu_qsar_predictions.csv")
    alerts = pd.read_csv(TABLE_DIR / "08_pains_brenk_alerts.csv")
    confs = pd.read_csv(TABLE_DIR / "08_3d_conformers.csv")
    return plant_tab, dl, ct, overlap, hubs, enrich, rank, pca, qsar_m, qsar_p, alerts, confs


plant_tab, dl, ct, overlap, hubs, enrich, rank, pca, qsar_m, qsar_p, alerts, confs = load_tables()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌱 Tanaman & Drug-likeness",
    "🎯 Target & Diabetes Tipe 2",
    "🕸️ Jaringan Interaktif & Hub",
    "🧬 Jalur Biologis",
    "🏆 Ruang Kimia & Peringkat",
    "🤖 QSAR DPP4, PAINS & 3D",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tanaman Jamu", f"{plant_tab['plant_id'].nunique()}")
    c2.metric("Senyawa Tervalidasi", f"{dl['compound'].nunique()}")
    c3.metric("Lolos Aturan Lipinski", f"{int(dl['lipinski_pass'].sum())}/{len(dl)}")
    c4.metric("Gen Irisan T2D", f"{len(overlap)}")

    pilihan_tanaman = st.multiselect(
        "Filter Tanaman:", sorted(plant_tab["plant_id"].unique()), default=sorted(plant_tab["plant_id"].unique())
    )
    st.dataframe(plant_tab[plant_tab["plant_id"].isin(pilihan_tanaman)], use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Struktur 2D Senyawa Jamu (RDKit)")
        st.image(str(FIG_DIR / "02_compound_structures_grid.png"), use_container_width=True)
    with col_b:
        st.subheader("Profil Drug-likeness")
        st.image(str(FIG_DIR / "02_druglikeness_overview.png"), use_container_width=True)

    st.dataframe(dl, use_container_width=True, hide_index=True)

with tab2:
    col_a, col_b = st.columns([1.1, 1.3])
    with col_a:
        st.subheader("Irisan Target Jamu & Gen T2D")
        st.image(str(FIG_DIR / "04_target_t2d_overlap.png"), use_container_width=True)
    with col_b:
        st.subheader("Jumlah Target Manusia per Senyawa (ChEMBL)")
        st.image(str(FIG_DIR / "03_targets_per_compound.png"), use_container_width=True)

    st.subheader("Daftar Gen Irisan (Target Jamu ∩ Diabetes Tipe 2)")
    st.dataframe(overlap, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Jaringan Interaktif: Tanaman → Senyawa → Target → Diabetes Tipe 2")
    st.caption("Hijau = Tanaman • Oranye = Senyawa Jamu • Biru = Protein Target • Merah = Diabetes Tipe 2 (geser & zoom!).")
    html_net = (FIG_DIR / "05_jamunet_interactive.html").read_text(encoding="utf-8")
    components.html(html_net, height=640)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.image(str(FIG_DIR / "05_network_overview.png"), use_container_width=True)
    with col_b:
        st.subheader("Top Hub Genes di Jaringan PPI STRING")
        st.dataframe(hubs.head(20), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Jalur Biologis Terkait (g:Profiler)")
    st.image(str(FIG_DIR / "06_pathway_enrichment.png"), use_container_width=True)
    st.dataframe(enrich.head(30), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Peringkat Kandidat Senyawa Jamu (Combined Score vs MedChem Lead Score)")
    st.caption(
        "Combined Score = 0.45×Network + 0.30×QED + 0.25×Tanimoto • "
        "MedChem Lead Score = Skor Terkalibrasi Log-Network + QED + Tanimoto Absolut − Penalti PAINS/Brenk"
    )
    sort_col = st.radio(
        "Urutkan Peringkat Berdasarkan:",
        ["medchem_lead_score", "combined_score"],
        index=0,
        horizontal=True,
    )
    st.dataframe(rank.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Peta Ruang Kimia Interaktif (PCA Morgan Fingerprint)")
    fig_pca = px.scatter(
        pca, x="PC1", y="PC2", color="group", symbol="role",
        hover_name="name", height=520,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with tab6:
    st.subheader("Evaluasi Model QSAR ML Target DPP4 (Murcko Scaffold Split)")
    st.dataframe(qsar_m, use_container_width=True, hide_index=True)
    st.image(str(FIG_DIR / "08_qsar_pains_summary.png"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prediksi QSAR DPP4 & Status PAINS Senyawa Jamu")
        st.dataframe(qsar_p, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Konformer 3D Siap-Docking (ETKDGv3 + MMFF94)")
        st.dataframe(confs, use_container_width=True, hide_index=True)
        st.subheader("Detail Filter PAINS & Brenk")
        st.dataframe(alerts, use_container_width=True, hide_index=True)
