"""
vortex_bridge.py — Modul Pembandingan Lintas-Model (Python IBM vs Vortex 10) & Validasi Historis.

PENTING (Etika & Akurasi Ilmiah):
Software eksternal Vortex 10 (SCTI) tidak terpasang di mesin ini. Oleh karena itu:
1. Modul ini TIDAK PERNAH memalsukan output Vortex 10.
2. Modul ini mengekspor spesifikasi parameter & template input yang selaras penuh
   untuk Vortex 10 (`results/vortex/vortex_scenario_spec.csv` & `vortex_project_template.xml`),
   serta menyediakan fungsi `import_vortex_csv()` bila pengguna sudah menjalankan
   Vortex 10 secara mandiri.
3. Sebagai pembanding struktural sementara (cross-model structural benchmark), modul ini
   menyandingkan jangkar skenario 2015 PHVA (N0=60) terhadap simulasi Python IBM kita
   serta menandai status Vortex secara transparan sebagai "Template Ready / Pending External Run".
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from engines.demographic import DemographicParams
from engines.simulator import run_scenario_montecarlo
from models.parameters import ScenarioConfig


def export_vortex_spec_and_xml(
    scenarios: dict[str, ScenarioConfig],
    demo_params: DemographicParams,
    out_dir: Path,
) -> Path:
    """Mengekspor spesifikasi parameter setara Vortex 10 & template XML dasar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    root = ET.Element("VortexProjectTemplate", name="NusantaraRhino_Vortex10_Match")

    for key, sc in scenarios.items():
        rows.append({
            "Scenario": key,
            "Vortex_Populations": sc.n_sites,
            "Initial_N_Pop1": sc.initial_n_site1,
            "Carrying_Capacity_Pop1": sc.k_site1,
            "Carrying_Capacity_Pop2": sc.k_site2,
            "Female_Breeding_Age": demo_params.female_maturity_age,
            "Male_Breeding_Age": demo_params.male_maturity_age,
            "Max_Age": demo_params.max_age,
            "Max_Progeny_Per_Brood": 1,
            "Pct_Adult_Females_Breeding_Approx": round(100.0 * demo_params.annual_conception_prob / 1.5, 1),
            "Calf_Mortality_Pct": round((1.0 - demo_params.calf_survival) * 100.0, 2),
            "Adult_Mortality_Pct": round((1.0 - demo_params.adult_survival) * 100.0, 2),
            "Lethal_Equivalents_2B": round(2.0 * demo_params.lethal_equivalents_B, 2),
            "Catastrophe_Prob_Pct": round(sc.catastrophe_prob * 100.0, 2),
            "Catastrophe_Survival_Multiplier": round(1.0 - sc.catastrophe_severity, 2),
            "Inter_Site_Catastrophe_Corr": sc.catastrophe_corr,
        })
        sc_el = ET.SubElement(root, "Scenario", id=key)
        for k, v in rows[-1].items():
            ET.SubElement(sc_el, k).text = str(v)

    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(out_dir / "vortex_scenario_spec.csv", index=False)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_dir / "vortex_project_template.xml", encoding="utf-8", xml_declaration=True)
    return out_dir / "vortex_scenario_spec.csv"


def run_historical_consistency_check(demo_params: DemographicParams, seed: int = 2026) -> pd.DataFrame:
    """
    Validasi Historis (1970 -> 2020, ~50 tahun):
    Secara historis Badak Jawa di Ujung Kulon tumbuh perlahan dari ~25-35 individu (akhir 1960-an)
    ke ~58-61 (2015 PHVA) hingga kisaran ~70-an (awal 2020-an) di bawah perlindungan ketat.
    Uji apakah parameter demografi dasar kita menghasilkan lintasan 50 tahun yang konsisten
    secara biologis (tanpa ledakan tak realistis maupun kepunahan otomatis tanpa bencana besar).
    """
    hist_cfg = ScenarioConfig(
        label="Historical Window Check (1970->2020, N0=30, K=80)",
        n_sites=1,
        initial_n_site1=30,
        k_site1=80,
        chronic_survival_drag=0.002,
        management_surv_boost=0.005,
        catastrophe_prob=0.005,
        catastrophe_severity=0.25,
        catastrophe_corr=0.0,
        founding_year=None,
        founder_size=0,
        k_site2=0,
    )
    _, rep_df = run_scenario_montecarlo(
        "Historical_1970_2020", hist_cfg, demo_params, horizon_years=50, n_replicates=120, seed=seed, record_yearly=False
    )
    return pd.DataFrame([{
        "Benchmark": "Ujung Kulon Historical Growth Window (N0=30 -> 50 yrs)",
        "Observed_Modern_Range": "58 - 80 individuals (2015-2023)",
        "Python_IBM_Mean_N50": round(float(rep_df["final_n"].mean()), 1),
        "Python_IBM_IQR": f"{np.percentile(rep_df['final_n'], 25):.1f} - {np.percentile(rep_df['final_n'], 75):.1f}",
        "Python_IBM_P5_P95": f"{np.percentile(rep_df['final_n'], 5):.1f} - {np.percentile(rep_df['final_n'], 95):.1f}",
        "Consistent_With_History": bool(45 <= rep_df["final_n"].mean() <= 85),
    }])


def build_cross_model_comparison_table(
    python_summary: pd.DataFrame,
    demo_params: DemographicParams,
    vortex_csv_path: Path | None = None,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Menyusun Table 6 (Perbandingan Python IBM vs Vortex 10).
    - Jika `vortex_csv_path` (ekspor nyata dari Vortex 10) ada, baca angka Vortex dari file tersebut.
    - Jika belum ada (Vortex 10 belum dijalankan eksternal), jalankan juga jangkar N0=60 (CPSG 2015)
      di Python IBM dan catat secara jujur bahwa kolom Vortex berstatus `Pending External Vortex 10 Run`.
    """
    anchor_cfg = ScenarioConfig(
        label="2015 PHVA Anchor (N0=60, K=80)",
        n_sites=1,
        initial_n_site1=60,
        k_site1=80,
        chronic_survival_drag=0.008,
        management_surv_boost=0.0,
        catastrophe_prob=0.02,
        catastrophe_severity=0.35,
        catastrophe_corr=0.0,
        founding_year=None,
        founder_size=0,
        k_site2=0,
    )
    _, anchor_reps = run_scenario_montecarlo(
        "A_2015_Anchor_N60", anchor_cfg, demo_params, horizon_years=100, n_replicates=150, seed=seed, record_yearly=False
    )

    vortex_data = {}
    if vortex_csv_path and vortex_csv_path.exists():
        vdf = pd.read_csv(vortex_csv_path)
        for _, r in vdf.iterrows():
            vortex_data[r["Scenario"]] = r.to_dict()

    rows = []
    for _, r in python_summary.iterrows():
        sc = r["Scenario"]
        vx = vortex_data.get(sc)
        rows.append({
            "Scenario": sc,
            "Python_IBM_Mean_Final_N": r["Mean_Final_N"],
            "Python_IBM_Extinction_Pct": r["Extinction_Prob_Pct"],
            "Python_IBM_Final_He": r["Mean_Final_He"],
            "Vortex_10_Status": "Imported from CSV" if vx else "Template Ready (Vortex 10 binary not installed locally)",
            "Vortex_10_Extinction_Pct": vx.get("Extinction_Prob_Pct", "Pending") if vx else "Pending",
            "Vortex_10_Mean_Final_N": vx.get("Mean_Final_N", "Pending") if vx else "Pending",
        })

    rows.append({
        "Scenario": "A_2015_Anchor_N60 (CPSG 2015 Baseline Sensitivity)",
        "Python_IBM_Mean_Final_N": round(float(anchor_reps["final_n"].mean()), 2),
        "Python_IBM_Extinction_Pct": round(float(anchor_reps["extinct"].mean() * 100.0), 2),
        "Python_IBM_Final_He": round(float(anchor_reps["final_he"].mean()), 4),
        "Vortex_10_Status": "CPSG 2015 Reference Anchor (see PHVA 2015 PDF)",
        "Vortex_10_Extinction_Pct": "Compare externally in Vortex 10",
        "Vortex_10_Mean_Final_N": "Compare externally in Vortex 10",
    })
    return pd.DataFrame(rows)
