"""
app.py
======
Dashboard interaktif Streamlit untuk Project 2: NusantaraBarcode.
Menampilkan:
  1. Eksplorasi Dataset Sekuens COI Ikan Konsumsi Nusantara
  2. Analisis Barcode Gap & Jarak Genetik K2P
  3. Pohon Filogenetik (Maximum-Likelihood IQ-TREE 2 & Neighbor-Joining)
  4. Peta Sebaran Sampel Geografis
  5. Identifikasi Spesies Bebas-Alignment (6-mer + ML & Studi Ablasi Audit GenBank)
  6. Genetika Populasi, Verifikasi ORF Mitokondria, & Audit Presisi GenBank
"""

from importlib import import_module
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

ml_mod = import_module("05_ml_species_id")

FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"
MODEL_PATH = PROJECT_DIR / "results" / "models" / "species_id_model.joblib"
FASTA_RAW = PROJECT_DIR / "data" / "processed" / "coi_raw.fasta"

st.set_page_config(
    page_title="NusantaraBarcode | DNA Barcoding & Filogeni",
    page_icon="🐟",
    layout="wide",
)

st.title("🐟 NusantaraBarcode: DNA Barcoding & Filogeni Ikan Konsumsi Indonesia")
st.caption("Bioinformatika Biodiversitas • Gen COI Mitokondria • MAFFT + IQ-TREE 2 + K2P + 6-mer ML")


@st.cache_data
def load_data():
    meta = pd.read_csv(TABLE_DIR / "02_aligned_metadata.csv")
    gap = pd.read_csv(TABLE_DIR / "03_barcode_gap_per_species.csv")
    dist_sum = pd.read_csv(TABLE_DIR / "03_distance_summary.csv")
    ml_comp = pd.read_csv(TABLE_DIR / "05_ml_comparison.csv")
    ablation_path = TABLE_DIR / "05_ml_ablation_audit.csv"
    ml_ablation = pd.read_csv(ablation_path) if ablation_path.exists() else ml_comp
    pop_gen = pd.read_csv(TABLE_DIR / "03b_population_genetics.csv")
    codon_st = pd.read_csv(TABLE_DIR / "03b_codon_position_stats.csv")
    gb_audit = pd.read_csv(TABLE_DIR / "03b_genbank_audit.csv")
    mono = pd.read_csv(TABLE_DIR / "03b_monophyly_check.csv")
    return meta, gap, dist_sum, ml_comp, ml_ablation, pop_gen, codon_st, gb_audit, mono


meta, gap, dist_sum, ml_comp, ml_ablation, pop_gen, codon_st, gb_audit, mono = load_data()
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ringkasan Dataset",
    "📏 Barcode Gap K2P",
    "🌳 Pohon Filogenetik",
    "🗺️ Peta Sampel",
    "🔬 Identifikasi Spesies (ML)",
    "🧬 Genetika Populasi & Audit",
])

with tab1:
    n_seq = len(meta)
    n_sp = meta["species"].nunique()
    n_id = int((meta["country"].str.lower() == "indonesia").sum())
    best_acc = float(ml_comp["CV_Accuracy"].iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sekuens Terkurasi", f"{n_seq}")
    c2.metric("Jumlah Spesies Ikan", f"{n_sp}")
    c3.metric("Sampel Asal Indonesia", f"{n_id}")
    c4.metric("Akurasi ML (6-mer CV)", f"{best_acc * 100:.1f}%")

    fam_filter = st.multiselect(
        "Filter Famili:", sorted(meta["family"].unique()), default=sorted(meta["family"].unique())
    )
    st.dataframe(meta[meta["family"].isin(fam_filter)], use_container_width=True, hide_index=True)

with tab2:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Ringkasan Jarak K2P")
        st.dataframe(dist_sum, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Barcode Gap Interaktif")
        fig_gap = px.scatter(
            gap,
            x="max_intra_k2p",
            y="min_inter_k2p",
            color="family",
            hover_name="species",
            hover_data=["n_seq", "nearest_neighbor", "barcode_gap"],
            color_discrete_map={"Lutjanidae": "#d62728", "Epinephelidae": "#2ca02c", "Scombridae": "#1f77b4"},
        )
        max_lim = float(max(gap["max_intra_k2p"].max(), gap["min_inter_k2p"].max()) * 1.1)
        fig_gap.add_shape(type="line", x0=0, y0=0, x1=max_lim, y1=max_lim, line=dict(dash="dash", color="black"))
        st.plotly_chart(fig_gap, use_container_width=True)

    st.image(str(FIG_DIR / "03_barcode_gap_plot.png"), use_container_width=True)
    st.dataframe(gap, use_container_width=True, hide_index=True)

with tab3:
    pilihan_pohon = st.radio(
        "Pilih Pohon Filogenetik:",
        [
            "Maximum-Likelihood (IQ-TREE 2 — Tingkat Spesies)",
            "Maximum-Likelihood (IQ-TREE 2 — Semua Sampel)",
            "Neighbor-Joining (K2P — Semua Sampel)",
        ],
        horizontal=True,
    )
    if "Tingkat Spesies" in pilihan_pohon and (FIG_DIR / "04_species_tree_ml.png").exists():
        st.image(str(FIG_DIR / "04_species_tree_ml.png"), use_container_width=True)
    elif "Semua Sampel" in pilihan_pohon and (FIG_DIR / "04_full_tree_ml.png").exists():
        st.image(str(FIG_DIR / "04_full_tree_ml.png"), use_container_width=True)
    else:
        st.image(str(FIG_DIR / "04_nj_full_tree.png"), use_container_width=True)

with tab4:
    st.subheader("Sebaran Geografis Sampel COI")
    map_html = (FIG_DIR / "06_sampling_map.html").read_text(encoding="utf-8")
    components.html(map_html, height=560)
    st.image(str(FIG_DIR / "06_country_distribution.png"), use_container_width=True)

with tab5:
    st.subheader("Performa Model Klasifikasi Bebas-Alignment (6-mer) & Ablasi Audit GenBank")
    c1, c2 = st.columns([1.1, 1.2])
    with c1:
        st.dataframe(ml_comp, use_container_width=True, hide_index=True)
        st.caption("Studi Ablasi: Sebelum vs Sesudah Pembersihan Mislabel GenBank")
        st.dataframe(ml_ablation, use_container_width=True, hide_index=True)
    with c2:
        st.image(str(FIG_DIR / "05_confusion_matrix.png"), use_container_width=True)

    st.divider()
    st.subheader("🧬 Uji Identifikasi Sekuens DNA COI")
    st.caption("Tempel sekuens DNA COI (format teks atau FASTA) untuk menebak spesies ikan secara instan.")
    default_seq = ""
    if FASTA_RAW.exists():
        lines = FASTA_RAW.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2:
            default_seq = lines[1][:600]
    seq_input = st.text_area("Sekuens DNA (A/C/G/T):", value=default_seq, height=140)
    if st.button("Identifikasi Spesies"):
        cleaned = "".join(
            l.strip() for l in seq_input.splitlines() if not l.strip().startswith(">")
        ).upper()
        if len(cleaned) < 50:
            st.warning("Sekuens terlalu pendek (< 50 bp).")
        elif bundle is None:
            st.info(
                "Artefak model `species_id_model.joblib` belum tersedia di folder lokal. "
                "Jalankan `python src/05_ml_species_id.py` terlebih dahulu untuk melatih model 6-mer."
            )
        else:
            res = ml_mod.identifikasi_sekuens(cleaned, bundle)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Prediksi Random Forest (6-mer):**")
                for sp, pr in res["top_rf"]:
                    st.write(f"- *{sp}* : **{pr * 100:.1f}%**")
            with col_b:
                st.markdown("**Sekuens Referensi Terdekat (Cosine Similarity):**")
                st.dataframe(pd.DataFrame(res["nearest_matches"]), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("Statistik Posisi Kodon, Verifikasi ORF Mitokondria, & Keragaman Sepanjang COI")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(codon_st, use_container_width=True, hide_index=True)
        st.metric("Spesies Monofiletik (IQ-TREE)", f"{int(mono['monophyletic'].sum())}/{len(mono)}")
    with c2:
        st.image(str(FIG_DIR / "03b_diversity_codon_stats.png"), use_container_width=True)

    st.subheader("Audit Presisi Anomali & Mislabel GenBank")
    st.dataframe(gb_audit, use_container_width=True, hide_index=True)
    st.subheader("Statistik Genetika Populasi per Spesies (π, Hd, Tajima's D)")
    st.dataframe(pop_gen, use_container_width=True, hide_index=True)
