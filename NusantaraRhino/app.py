"""
app.py — Dashboard Riset Interaktif NusantaraRhino (Streamlit + Plotly).

Menampilkan:
  1. Lintasan Populasi & Probabilitas Kepunahan (dengan interval ketidakpastian P5–P95)
  2. Dinamika Genetika Populasi (Ho, He, Allelic Richness, F_IBD vs Teori Wright-Fisher)
  3. Eksperimen Sensitivitas (Demografi PRCC, Genetika B x H0, Korelasi Bencana rho, K, Robustness)
  4. Validasi Historis, Audit Provenance Parameter, & Jembatan Pembanding Vortex 10
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

st.set_page_config(
    page_title="NusantaraRhino | Integrated Javan Rhino PVA",
    page_icon="🦏",
    layout="wide",
)

st.title("🦏 NusantaraRhino: Integrated Demographic, Genetic & Environmental PVA")
st.caption(
    "Evaluating the Long-Term Viability of the Javan Rhinoceros (Rhinoceros sondaicus) • "
    "Individual-Based Stochastic Modeling (100 Years, 250 Monte Carlo Replicates/Scenario)"
)


@st.cache_data
def load_all():
    traj = pd.read_csv(TABLE_DIR / "yearly_trajectories_summary.csv")
    t1 = pd.read_csv(TABLE_DIR / "Table1_Parameters_and_Sources.csv")
    t2 = pd.read_csv(TABLE_DIR / "Table2_Scenario_Configurations.csv")
    t3 = pd.read_csv(TABLE_DIR / "Table3_Demographic_Outcomes.csv")
    t4 = pd.read_csv(TABLE_DIR / "Table4_Genetic_Outcomes.csv")
    t5 = pd.read_csv(TABLE_DIR / "Table5_Sensitivity_PRCC.csv")
    t6 = pd.read_csv(TABLE_DIR / "Table6_Python_vs_Vortex.csv")
    wf = pd.read_csv(TABLE_DIR / "wright_fisher_baselines.csv")
    exp2 = pd.read_csv(TABLE_DIR / "exp2_genetic_sensitivity.csv")
    exp3 = pd.read_csv(TABLE_DIR / "exp3_catastrophe_correlation.csv")
    exp4 = pd.read_csv(TABLE_DIR / "exp4_carrying_capacity.csv")
    exp5 = pd.read_csv(TABLE_DIR / "exp5_combined_robustness.csv")
    hist = pd.read_csv(TABLE_DIR / "validation_historical_window.csv")
    evid = pd.read_csv(ROOT / "data" / "empirical_genetic_evidence.csv")
    return traj, t1, t2, t3, t4, t5, t6, wf, exp2, exp3, exp4, exp5, hist, evid


traj, t1, t2, t3, t4, t5, t6, wf, exp2, exp3, exp4, exp5, hist, evid = load_all()

COLORS = {
    "A_Baseline_UKNP": "#d62728",
    "B_Managed_UKNP": "#2ca02c",
    "C_Metapop_Uncorrelated": "#1f77b4",
    "C_Metapop_Correlated": "#9467bd",
    "D_Integrated_Managed_Metapop": "#ff7f0e",
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Skenario & Trajektori",
    "🧬 Genetika Populasi",
    "🔬 5 Eksperimen Sensitivitas",
    "🛡️ Validasi & Vortex 10 Bridge",
    "📚 Provenance Parameter & Bukti",
])

with tab1:
    st.info(
        "Catatan ilmiah: Model memperkirakan risiko bersyarat di bawah asumsi dan ketidakpastian eksplisit — "
        "bukan prediksi tahun pasti kepunahan."
    )
    n_cols = max(1, len(t3))
    cols = st.columns(n_cols)
    for idx, (_, row) in enumerate(t3.iterrows()):
        with cols[idx % n_cols]:
            st.metric(
                row["Scenario"].replace("_", " "),
                f"N = {row['Mean_Final_N']}",
                f"Quasi-Ext: {row['Quasi_Extinction_Prob_Pct']}%",
                delta_color="inverse",
            )

    fig_traj = go.Figure()
    for sc, col in COLORS.items():
        sub = traj[traj["scenario"] == sc]
        fig_traj.add_trace(go.Scatter(
            x=sub["year"], y=sub["p95_n"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_traj.add_trace(go.Scatter(
            x=sub["year"], y=sub["p5_n"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(128,128,128,0.12)", showlegend=False, hoverinfo="skip"
        ))
        fig_traj.add_trace(go.Scatter(
            x=sub["year"], y=sub["mean_n"], mode="lines", name=sc, line=dict(color=col, width=3)
        ))
    fig_traj.add_hline(y=15, line_dash="dash", annotation_text="Quasi-Extinction (N=15)")
    fig_traj.update_layout(title="Lintasan Populasi Global (Mean + P5–P95)", xaxis_title="Tahun", yaxis_title="Total Individu")
    st.plotly_chart(fig_traj, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Table 3: Demographic Outcomes (Tahun ke-100)")
        st.dataframe(t3, use_container_width=True, hide_index=True)
    with c2:
        st.image(str(FIG_DIR / "Figure3_Extinction_Curves.png"), use_container_width=True)

with tab2:
    st.subheader("Retensi Heterozigositas (He) & Akumulasi Inbreeding Pedigree (F_IBD)")
    c1, c2 = st.columns(2)
    with c1:
        fig_he = px.line(traj, x="year", y="mean_he", color="scenario", color_discrete_map=COLORS,
                         title="Expected Heterozygosity (He) vs Waktu")
        fig_he.add_scatter(x=wf["year"], y=wf["WF_Ne25"], mode="lines", name="Wright-Fisher (Ne=25)", line=dict(dash="dash", color="black"))
        st.plotly_chart(fig_he, use_container_width=True)
    with c2:
        fig_f = px.line(traj, x="year", y="mean_f", color="scenario", color_discrete_map=COLORS,
                        title="Rata-rata Koefisien Inbreeding IBD (F_IBD)")
        st.plotly_chart(fig_f, use_container_width=True)

    st.subheader("Table 4: Genetic Outcomes (Tahun ke-100)")
    st.dataframe(t4, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Eksperimen 1: Sensitivitas Demografi (Spearman PRCC)")
    st.dataframe(t5, use_container_width=True, hide_index=True)
    st.image(str(FIG_DIR / "Figure5_Catastrophe_Correlation_Sensitivity.png"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Eksperimen 2: Sensitivitas Genetika (B × H0)")
        st.dataframe(exp2, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Eksperimen 4: Daya Dukung Habitat (K)")
        st.dataframe(exp4, use_container_width=True, hide_index=True)

    st.subheader("Eksperimen 3 & 5: Korelasi Bencana (ρ) & Ketahanan Strategi")
    st.image(str(FIG_DIR / "Figure6_Integrated_Robustness.png"), use_container_width=True)
    st.dataframe(exp5, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Validasi Historis (1970–2020)")
    st.dataframe(hist, use_container_width=True, hide_index=True)

    st.subheader("Table 6: Cross-Model Comparison Bridge (Python IBM vs Vortex 10)")
    st.warning(
        "Software eksternal Vortex 10 belum dijalankan pada mesin ini. Template XML & tabel spesifikasi parameter "
        "telah diekspor ke `results/vortex/` untuk perbandingan independen yang setara tanpa fabrikasi output."
    )
    st.dataframe(t6, use_container_width=True, hide_index=True)
    st.image(str(FIG_DIR / "Figure1_Conceptual_Framework.png"), use_container_width=True)

with tab5:
    st.subheader("Table 1: Parameter Database & Provenance Audit")
    atype = st.multiselect(
        "Filter assumption_type:",
        sorted(t1["assumption_type"].unique()),
        default=sorted(t1["assumption_type"].unique()),
    )
    st.dataframe(t1[t1["assumption_type"].isin(atype)], use_container_width=True, hide_index=True)
    st.subheader("Lapisan 1: Inventaris Bukti Genetik & Literatur Empiris")
    st.dataframe(evid, use_container_width=True, hide_index=True)
    st.subheader("Table 2: Konfigurasi Skenario")
    st.dataframe(t2, use_container_width=True, hide_index=True)
