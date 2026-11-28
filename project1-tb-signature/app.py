"""
app.py
======
Dashboard interaktif Streamlit untuk Project 1: TB-Signature.
Menampilkan:
  1. Ringkasan Data & Kohort (Discovery GPL6947 vs External Validation GSE37250)
  2. Kontrol Kualitas (PCA) & Ekspresi Diferensial (Volcano Plot Interaktif + Cek Gen)
  3. Pengayaan Jalur Biologis (g:Profiler)
  4. Performa Machine Learning (Nested CV + Validasi Eksternal Single-Sample & Kohort) & Simulator 8-Gen
  5. Model Edukasional Boolean Network Imun TBC
  6. Strict LODO Cross-Cohort, Benchmark Sweeney 3-Gene Utuh, & Sistem ODE Kontinu
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
PROC_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"
MODEL_PATH = PROJECT_DIR / "results" / "models" / "tb_signature_model.joblib"

st.set_page_config(
    page_title="TB-Signature | Biomarker TBC & Machine Learning",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 TB-Signature: Penemuan Biomarker Darah TBC Berbasis Transcriptomics & ML")
st.caption(
    "Portofolio Bioinformatika & Biologi Komputasi • Discovery: GSE19444, GSE19439, GSE19442 (GPL6947) "
    "• Validasi Eksternal: GSE37250 (GPL10558)"
)


@st.cache_data
def load_tables():
    sample_sum = pd.read_csv(TABLE_DIR / "02_sample_summary.csv")
    pca_disc = pd.read_csv(TABLE_DIR / "02_pca_discovery.csv", index_col=0)
    de = pd.read_csv(TABLE_DIR / "03_differential_expression.csv")
    enrich = pd.read_csv(TABLE_DIR / "04_pathway_enrichment.csv")
    cv_comp = pd.read_csv(TABLE_DIR / "05_cv_comparison.csv")
    sig_genes = pd.read_csv(TABLE_DIR / "05_signature_genes.csv")
    ext_val = pd.read_csv(TABLE_DIR / "05_external_validation.csv")
    bool_att = pd.read_csv(TABLE_DIR / "06_boolean_attractors.csv")
    lodo = pd.read_csv(TABLE_DIR / "07_lodo_validation.csv")
    bench = pd.read_csv(TABLE_DIR / "07_benchmark_sweeney3.csv")
    ode_df = pd.read_csv(TABLE_DIR / "07_ode_simulation.csv")
    return sample_sum, pca_disc, de, enrich, cv_comp, sig_genes, ext_val, bool_att, lodo, bench, ode_df


@st.cache_data
def load_expr_subset(genes: tuple[str, ...]):
    parquet_path = PROC_DIR / "discovery_expr.parquet"
    meta_path = PROC_DIR / "discovery_meta.csv"
    if not parquet_path.exists() or not meta_path.exists():
        return None, None
    expr_disc = pd.read_parquet(parquet_path).loc[list(genes)]
    meta_disc = pd.read_csv(meta_path, index_col=0)
    return expr_disc, meta_disc


def predict_from_signature_table(x_vec: np.ndarray, sig_df: pd.DataFrame, bundle: dict | None) -> float:
    """Prediksi probabilitas Active TB dari model .joblib atau fallback eksak dari tabel koefisien."""
    if bundle is not None and "pipeline" in bundle:
        return float(bundle["pipeline"].predict_proba(x_vec)[0, 1])
    # Fallback clone-safe jika file .joblib belum dibangkitkan lokal
    coefs = sig_df["coef_logreg"].values
    intercept = float(sig_df["intercept_logreg"].iloc[0]) if "intercept_logreg" in sig_df.columns else -0.5
    mean = sig_df["scaler_mean"].values if "scaler_mean" in sig_df.columns else np.zeros(len(coefs))
    scale = sig_df["scaler_scale"].values if "scaler_scale" in sig_df.columns else np.ones(len(coefs))
    z = (x_vec.ravel() - mean) / (scale + 1e-8)
    logit = intercept + float(np.dot(coefs, z))
    return float(1.0 / (1.0 + np.exp(-logit)))


sample_sum, pca_disc, de, enrich, cv_comp, sig_genes, ext_val, bool_att, lodo, bench, ode_df = load_tables()
model_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
panel_genes = model_bundle["genes"] if model_bundle is not None else sig_genes["gene"].tolist()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ringkasan & Kohort",
    "🌋 Volcano & Cek Gen",
    "🧬 Jalur Biologis",
    "🤖 Model ML & Simulasi Pasien",
    "⚡ Boolean Network",
    "📈 LODO, Benchmark, & ODE",
])

with tab1:
    n_disc = int(sample_sum[sample_sum["dataset"] != "GSE37250"]["jumlah_sampel"].sum())
    n_val = int(sample_sum[sample_sum["dataset"] == "GSE37250"]["jumlah_sampel"].sum())
    n_sig = int((de["status"] != "Tidak Signifikan").sum())
    auc_all = float(ext_val.loc[ext_val["Evaluasi"] == "Active_TB vs Semua Non-TB (LTBI + Other)", "AUC"].iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampel Discovery", f"{n_disc}")
    c2.metric("Sampel Validasi Eksternal", f"{n_val}")
    c3.metric("Gen Signifikan (FDR < 0.05)", f"{n_sig}")
    c4.metric("AUC Validasi Eksternal", f"{auc_all:.3f}")

    st.subheader("Komposisi Sampel per Dataset")
    st.dataframe(sample_sum, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("PCA Interaktif Discovery Cohort")
        fig_pca = px.scatter(
            pca_disc.reset_index(),
            x="PC1", y="PC2", color="group", symbol="dataset",
            hover_data=["sample_id", "dataset", "group"],
            color_discrete_map={"Active_TB": "#d62728", "Latent_TB": "#ff7f0e", "Healthy": "#2ca02c"},
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    with col_b:
        st.subheader("Plot QC & PCA (Static)")
        st.image(str(FIG_DIR / "02_qc_pca.png"), use_container_width=True)

with tab2:
    st.subheader("Volcano Plot Interaktif (Discovery Cohort)")
    de_plot = de.copy()
    de_plot["neg_log10_fdr"] = -np.log10(np.clip(de_plot["fdr"], 1e-300, 1.0))
    fig_vol = px.scatter(
        de_plot,
        x="log2FC", y="neg_log10_fdr", color="status",
        hover_data=["gene", "log2FC", "fdr"],
        color_discrete_map={
            "Naik di TB (Up)": "#d62728",
            "Turun di TB (Down)": "#1f77b4",
            "Tidak Signifikan": "#cccccc",
        },
        opacity=0.75,
        height=480,
    )
    fig_vol.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=-0.5, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()
    st.subheader("🔍 Inspeksi Ekspresi Gen Individual")
    pilihan_default = sig_genes["gene"].iloc[0]
    all_genes = sorted(de["gene"].tolist())
    chosen_gene = st.selectbox("Pilih Gen:", all_genes, index=all_genes.index(pilihan_default))
    expr_sub, meta_sub = load_expr_subset((chosen_gene,))
    if expr_sub is not None and meta_sub is not None:
        df_box = meta_sub.copy()
        df_box["ekspresi_log2"] = expr_sub.loc[chosen_gene].values
        fig_box = px.box(
            df_box, x="group", y="ekspresi_log2", color="group", points="all",
            color_discrete_map={"Active_TB": "#d62728", "Latent_TB": "#ff7f0e", "Healthy": "#2ca02c"},
            title=f"Ekspresi Gen {chosen_gene} di Discovery Cohort",
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        g_row = de[de["gene"] == chosen_gene].iloc[0]
        st.info(
            f"Ringkasan `{chosen_gene}`: Mean Active TB = `{g_row['mean_active_tb']:.2f}`, "
            f"Mean Kontrol = `{g_row['mean_control']:.2f}`, log2FC = `{g_row['log2FC']:+.2f}`, "
            f"FDR = `{g_row['fdr']:.2e}`. (Jalankan `python run_all.py` untuk mengunduh matriks per sampel)."
        )
    st.image(str(FIG_DIR / "03_heatmap_top30.png"), caption="Heatmap Top 30 DEGs", use_container_width=True)

with tab3:
    st.subheader("Pengayaan Jalur Biologis (g:Profiler)")
    st.image(str(FIG_DIR / "04_pathway_enrichment.png"), use_container_width=True)
    st.dataframe(enrich.head(30), use_container_width=True, hide_index=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Perbandingan Model (5-Fold & Nested CV Discovery)")
        st.dataframe(cv_comp, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("8-Gen TB Signature (Discovery & External Importance)")
        st.dataframe(sig_genes, use_container_width=True, hide_index=True)

    st.subheader("Validasi Eksternal Lintas Platform (GSE37250, n=537 — Cohort vs Single-Sample)")
    st.dataframe(ext_val, use_container_width=True, hide_index=True)
    st.image(str(FIG_DIR / "05_roc_curves.png"), use_container_width=True)
    st.image(str(FIG_DIR / "05_signature_importance.png"), use_container_width=True)

    st.divider()
    st.subheader("🧪 Simulator Uji Signature 8-Gen (Edukasi)")
    st.caption("Geser nilai z-score relatif tiap gen pada sampel darah untuk melihat prediksi probabilitas Active TB oleh model.")
    sliders = {}
    cols = st.columns(4)
    ordered_sig_df = sig_genes.set_index("gene").loc[panel_genes].reset_index()
    for idx, g in enumerate(panel_genes):
        with cols[idx % 4]:
            sliders[g] = st.slider(g, min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
    x_vec = np.array([[sliders[g] for g in panel_genes]])
    prob = predict_from_signature_table(x_vec, ordered_sig_df, model_bundle)
    st.progress(prob, text=f"Probabilitas Active TB: {prob * 100:.1f}%")
    if prob >= 0.5:
        st.error(f"Prediksi: **Active TB** (skor = {prob:.3f}) — *hanya simulasi edukasi, bukan alat diagnosis klinis.*")
    else:
        st.success(f"Prediksi: **Non-TB** (skor = {prob:.3f}) — *hanya simulasi edukasi, bukan alat diagnosis klinis.*")

with tab5:
    st.subheader("Model Jaringan Boolean Respons Imun TBC (Edukasi)")
    st.info(
        "Ini adalah model kualitatif berbasis logika biner (0/1) untuk menunjukkan konsep "
        "attractor pada jalur sinyal Interferon-gamma — bukan model klinis kuantitatif."
    )
    st.dataframe(bool_att, use_container_width=True, hide_index=True)
    st.image(str(FIG_DIR / "06_boolean_network.png"), use_container_width=True)

with tab6:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Leave-One-Dataset-Out (LODO: Strict vs Fixed-8)")
        st.dataframe(lodo, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Benchmark vs Skor Literatur Sweeney 3-Gen Utuh (GSE37250)")
        st.dataframe(bench, use_container_width=True, hide_index=True)

    st.image(str(FIG_DIR / "07_benchmark_and_ode.png"), use_container_width=True)
    st.subheader("Eksplorasi Model Matematika Kontinu (Sistem ODE)")
    scen = st.selectbox("Pilih Skenario Imun:", sorted(ode_df["Skenario"].unique()))
    sub_ode = ode_df[ode_df["Skenario"] == scen].melt(
        id_vars=["Skenario", "Hari"], var_name="Variabel", value_name="Nilai"
    )
    fig_ode = px.line(sub_ode, x="Hari", y="Nilai", color="Variabel", title=f"Dinamika ODE: {scen}")
    st.plotly_chart(fig_ode, use_container_width=True)
