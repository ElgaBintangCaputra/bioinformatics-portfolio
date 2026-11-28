"""
run_experiments.py — Menjalankan seluruh simulasi utama (5 Skenario Faktorial A, B, C1, C2, D)
+ 5 Eksperimen Sensitivitas (termasuk Latin Hypercube Sampling & True PRCC)
& menghasilkan Table 1–6 untuk manuskrip NusantaraRhino.
"""

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, qmc, rankdata, spearmanr, t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from engines.demographic import DemographicParams
from engines.genetics import wright_fisher_curve
from engines.simulator import run_scenario_montecarlo
from models.parameters import ScenarioConfig, load_scenarios, load_validated_parameters
from validation.vortex_bridge import (
    build_cross_model_comparison_table,
    export_vortex_spec_and_xml,
    run_historical_consistency_check,
)

RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
VORTEX_DIR = RESULTS_DIR / "vortex"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
VORTEX_DIR.mkdir(parents=True, exist_ok=True)


def build_base_demo_params(param_map: dict) -> DemographicParams:
    return DemographicParams(
        female_maturity_age=int(param_map["female_maturity_age"].value),
        male_maturity_age=int(param_map["male_maturity_age"].value),
        max_reproductive_age=int(param_map["max_reproductive_age"].value),
        max_age=int(param_map["max_age"].value),
        # Gunakan ceil(3.5) = 4 langkah tahunan diskrit (3 tahun penuh blokir pasca-melahirkan)
        # agar interval antar-kelahiran selalu mematuhi batas biologis >= 3.5 tahun
        min_interbirth_years=int(math.ceil(param_map["min_interbirth_interval_years"].value)),
        sex_ratio_female=float(param_map["sex_ratio_at_birth"].value),
        calf_survival=float(param_map["calf_survival_annual"].value),
        subadult_survival=float(param_map["subadult_survival_annual"].value),
        adult_survival=float(param_map["adult_survival_annual"].value),
        senescent_survival=float(param_map["senescent_survival_annual"].value),
        annual_conception_prob=float(param_map["annual_conception_prob_receptive"].value),
        lethal_equivalents_B=float(param_map["lethal_equivalents_B"].value),
        initial_h0=float(param_map["initial_expected_heterozygosity_H0"].value),
        n_loci=int(param_map["n_unlinked_loci"].value),
        n_alleles=int(param_map["initial_alleles_per_locus"].value),
        env_sd_survival=float(param_map["environmental_sd_survival"].value),
    )


def wilson_ci_pct(k: int, n: int, z: float = 1.959963984540054) -> str:
    """Menghitung interval kepercayaan 95% Wilson score untuk proporsi binomial (dalam %)."""
    if n <= 0:
        return "0.0 - 0.0"
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return f"{lo:.1f} - {hi:.1f}"


def compute_prcc(df: pd.DataFrame, param_cols: list[str], target_col: str) -> dict[str, tuple[float, float]]:
    """
    Menghitung Partial Rank Correlation Coefficient (PRCC) sejati:
    1. Transformasi peringkat (rankdata) pada seluruh parameter input X_1..X_k dan luaran Y.
    2. Untuk setiap parameter X_j, regresi linier OLS rank(X_j) terhadap rank(X_{-j}) -> residual e_X,
       dan regresi linier OLS rank(Y) terhadap rank(X_{-j}) -> residual e_Y.
    3. PRCC(X_j, Y) = korelasi Pearson(e_X, e_Y), dengan uji-t derajat bebas df = N - 2 - (k - 1).
    """
    n = len(df)
    k = len(param_cols)
    ranked_X = np.column_stack([rankdata(df[c].values) for c in param_cols])
    ranked_y = rankdata(df[target_col].values)

    results = {}
    df_dof = max(1, n - 2 - (k - 1))
    for j, col in enumerate(param_cols):
        other_idx = [i for i in range(k) if i != j]
        Z = np.column_stack([np.ones(n), ranked_X[:, other_idx]])
        beta_x, _, _, _ = np.linalg.lstsq(Z, ranked_X[:, j], rcond=None)
        res_x = ranked_X[:, j] - Z @ beta_x

        beta_y, _, _, _ = np.linalg.lstsq(Z, ranked_y, rcond=None)
        res_y = ranked_y - Z @ beta_y

        r_val = float(pearsonr(res_x, res_y)[0])
        r_clip = max(-0.999999, min(0.999999, r_val))
        t_stat = abs(r_clip) * math.sqrt(df_dof / max(1e-12, 1.0 - r_clip * r_clip))
        p_val = float(2.0 * (1.0 - student_t.cdf(t_stat, df=df_dof)))
        results[col] = (r_val, p_val)
    return results


def summarize_yearly(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung mean, median, dan interval P5–P95 serta kurva kumulatif kepunahan."""
    def q5(x): return float(np.percentile(x, 5))
    def q95(x): return float(np.percentile(x, 95))

    agg = (
        yearly_df.groupby(["scenario", "year"], as_index=False)
        .agg(
            mean_n=("n_total", "mean"),
            median_n=("n_total", "median"),
            p5_n=("n_total", q5),
            p95_n=("n_total", q95),
            mean_he=("he", "mean"),
            p5_he=("he", q5),
            p95_he=("he", q95),
            mean_ho=("ho", "mean"),
            mean_ar=("allelic_richness", "mean"),
            mean_f=("mean_f_ibd", "mean"),
            p_extinct_yr=("n_total", lambda s: float(np.mean(s == 0) * 100.0)),
            p_quasi_yr=("n_total", lambda s: float(np.mean(s <= 15) * 100.0)),
        )
    )
    return agg


def run_exp1_demographic_prcc(
    base_sc: ScenarioConfig, base_dp: DemographicParams, seed: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Eksperimen 1: Demographic Sensitivity menggunakan Latin Hypercube Sampling (scipy.stats.qmc.LatinHypercube)
    & Partial Rank Correlation Coefficient (PRCC) sejati + Spearman bivariat.
    """
    n_samples = 36
    sampler = qmc.LatinHypercube(d=5, seed=seed + 101)
    u = sampler.random(n=n_samples)

    # Batas bawah & atas parameter untuk LHS:
    # [adult_survival, calf_survival, annual_conception_prob, catastrophe_prob, carrying_capacity_K]
    l_bounds = [0.93, 0.76, 0.30, 0.005, 60.0]
    u_bounds = [0.99, 0.94, 0.60, 0.040, 115.0]
    scaled = qmc.scale(u, l_bounds, u_bounds)

    samples = []
    for i in range(n_samples):
        adult_s = float(scaled[i, 0])
        calf_s = float(scaled[i, 1])
        concep = float(scaled[i, 2])
        cat_p = float(scaled[i, 3])
        k_val = int(round(scaled[i, 4]))

        dp = DemographicParams(**{
            **base_dp.__dict__,
            "adult_survival": adult_s,
            "calf_survival": calf_s,
            "annual_conception_prob": concep,
        })
        sc = ScenarioConfig(**{**base_sc.model_dump(), "k_site1": k_val, "catastrophe_prob": cat_p})
        _, reps = run_scenario_montecarlo(
            f"Exp1_{i}", sc, dp, horizon_years=100, n_replicates=40, seed=seed + i, record_yearly=False
        )
        samples.append({
            "adult_survival": adult_s,
            "calf_survival": calf_s,
            "annual_conception_prob": concep,
            "catastrophe_prob": cat_p,
            "carrying_capacity_K": k_val,
            "extinction_prob_pct": float(reps["extinct"].mean() * 100.0),
            "quasi_ext_prob_pct": float(reps["quasi_extinct"].mean() * 100.0),
            "mean_final_n": float(reps["final_n"].mean()),
        })

    df_samples = pd.DataFrame(samples)
    param_cols = [
        "adult_survival",
        "calf_survival",
        "annual_conception_prob",
        "catastrophe_prob",
        "carrying_capacity_K",
    ]
    prcc_q = compute_prcc(df_samples, param_cols, "quasi_ext_prob_pct")
    prcc_n = compute_prcc(df_samples, param_cols, "mean_final_n")

    sens_rows = []
    for col in param_cols:
        r_ext, p_ext = spearmanr(df_samples[col], df_samples["quasi_ext_prob_pct"])
        r_n, p_n = spearmanr(df_samples[col], df_samples["mean_final_n"])
        pq_val, pq_p = prcc_q[col]
        pn_val, pn_p = prcc_n[col]
        sens_rows.append({
            "Parameter": col,
            "PRCC_QuasiExt": round(float(pq_val), 4),
            "p_val_PRCC_QuasiExt": round(float(pq_p), 4),
            "PRCC_FinalN": round(float(pn_val), 4),
            "p_val_PRCC_FinalN": round(float(pn_p), 4),
            "Spearman_r_QuasiExt": round(float(r_ext), 4),
            "p_val_QuasiExt": round(float(p_ext), 4),
            "Spearman_r_FinalN": round(float(r_n), 4),
            "p_val_FinalN": round(float(p_n), 4),
        })
    sens_df = pd.DataFrame(sens_rows).sort_values("PRCC_QuasiExt", key=lambda s: s.abs(), ascending=False)
    return df_samples, sens_df


def run_exp2_genetics(base_sc: ScenarioConfig, base_dp: DemographicParams, seed: int = 2026) -> pd.DataFrame:
    """Eksperimen 2: Sensitivitas Genetika (Lethal Equivalents B x Heterozigositas Awal H0)."""
    rows = []
    for b_val in [0.0, 1.57, 3.14]:
        for h0_val in [0.35, 0.55, 0.75]:
            dp = DemographicParams(**{**base_dp.__dict__, "lethal_equivalents_B": b_val, "initial_h0": h0_val})
            _, reps = run_scenario_montecarlo(
                f"Exp2_B{b_val}_H{h0_val}",
                base_sc,
                dp,
                horizon_years=100,
                n_replicates=80,
                seed=seed + int(b_val * 10 + h0_val * 100),
                record_yearly=False,
            )
            rows.append({
                "Lethal_Equivalents_B": b_val,
                "Initial_H0": h0_val,
                "Mean_Final_N": round(float(reps["final_n"].mean()), 2),
                "Quasi_Ext_Pct": round(float(reps["quasi_extinct"].mean() * 100.0), 2),
                "Mean_Final_He": round(float(reps["final_he"].mean()), 4),
                "Mean_Final_F_IBD": round(float(reps["final_f_ibd"].mean()), 4),
            })
    return pd.DataFrame(rows)


def run_exp3_catastrophe_correlation(
    sc_c: ScenarioConfig, base_dp: DemographicParams, seed: int = 2026
) -> pd.DataFrame:
    """Eksperimen 3: Menguji P(C1, C2) != P(C1)*P(C2) dengan rho in {0.0, 0.3, 0.6, 0.9} & keparahan."""
    rows = []
    for rho in [0.0, 0.3, 0.6, 0.9]:
        for sev in [0.25, 0.35, 0.50]:
            sc = ScenarioConfig(**{**sc_c.model_dump(), "catastrophe_corr": rho, "catastrophe_severity": sev})
            _, reps = run_scenario_montecarlo(
                f"Exp3_rho{rho}_sev{sev}",
                sc,
                base_dp,
                horizon_years=100,
                n_replicates=100,
                seed=seed + int(rho * 100 + sev * 100),
                record_yearly=False,
            )
            rows.append({
                "Catastrophe_Corr_rho": rho,
                "Severity": sev,
                "Simultaneous_Catastrophes_Mean": round(float(reps["simultaneous_cat_events"].mean()), 2),
                "Extinction_Prob_Pct": round(float(reps["extinct"].mean() * 100.0), 2),
                "Quasi_Ext_Prob_Pct": round(float(reps["quasi_extinct"].mean() * 100.0), 2),
                "Mean_Final_N": round(float(reps["final_n"].mean()), 2),
                "Mean_Final_He": round(float(reps["final_he"].mean()), 4),
            })
    return pd.DataFrame(rows)


def run_exp4_carrying_capacity(base_sc: ScenarioConfig, base_dp: DemographicParams, seed: int = 2026) -> pd.DataFrame:
    """Eksperimen 4: Daya Dukung Habitat Ujung Kulon K in {60, 80, 100, 120}."""
    rows = []
    for k_val in [60, 80, 100, 120]:
        sc = ScenarioConfig(**{**base_sc.model_dump(), "k_site1": k_val})
        _, reps = run_scenario_montecarlo(
            f"Exp4_K{k_val}",
            sc,
            base_dp,
            horizon_years=100,
            n_replicates=100,
            seed=seed + k_val,
            record_yearly=False,
        )
        rows.append({
            "Carrying_Capacity_K": k_val,
            "Mean_Final_N": round(float(reps["final_n"].mean()), 2),
            "Quasi_Ext_Prob_Pct": round(float(reps["quasi_extinct"].mean() * 100.0), 2),
            "Mean_Final_He": round(float(reps["final_he"].mean()), 4),
            "Mean_Final_F_IBD": round(float(reps["final_f_ibd"].mean()), 4),
        })
    return pd.DataFrame(rows)


def run_exp5_combined_robustness(
    scenarios: dict[str, ScenarioConfig], base_dp: DemographicParams, seed: int = 2026
) -> pd.DataFrame:
    """
    Eksperimen 5 (Novelty 3): Robustness kelima strategi (A, B, C1, C2, D) di bawah ketidakpastian
    kombinasi B (inbreeding load) dan probabilitas bencana.
    """
    rows = []
    regimes = [
        ("Optimis (B=0.5, Cat=0.01)", 0.5, 0.01),
        ("Moderat (B=1.57, Cat=0.02)", 1.57, 0.02),
        ("Pesimis (B=3.14, Cat=0.03)", 3.14, 0.03),
    ]
    for rname, b_val, c_prob in regimes:
        dp = DemographicParams(**{**base_dp.__dict__, "lethal_equivalents_B": b_val})
        for sc_key, sc in scenarios.items():
            sc_mod = ScenarioConfig(**{**sc.model_dump(), "catastrophe_prob": c_prob})
            _, reps = run_scenario_montecarlo(
                f"Exp5_{rname}_{sc_key}",
                sc_mod,
                dp,
                horizon_years=100,
                n_replicates=80,
                seed=seed + int(b_val * 10),
                record_yearly=False,
            )
            rows.append({
                "Uncertainty_Regime": rname,
                "Scenario": sc_key,
                "Mean_Final_N": round(float(reps["final_n"].mean()), 2),
                "Quasi_Ext_Prob_Pct": round(float(reps["quasi_extinct"].mean() * 100.0), 2),
                "Extinction_Prob_Pct": round(float(reps["extinct"].mean() * 100.0), 2),
                "Mean_Final_He": round(float(reps["final_he"].mean()), 4),
            })
    return pd.DataFrame(rows)


def main():
    print("== NusantaraRhino: Menjalankan Simulasi Utama (5 Skenario Faktorial) & 5 Eksperimen ==")
    param_map = load_validated_parameters(ROOT / "data" / "parameters.csv")
    raw_cfg, scenarios = load_scenarios(ROOT / "configs" / "scenarios.yaml")
    demo_params = build_base_demo_params(param_map)

    horizon = int(raw_cfg["horizon_years"])
    n_reps = int(raw_cfg["n_replicates"])
    seed = int(raw_cfg["random_seed"])

    pd.read_csv(ROOT / "data" / "parameters.csv").to_csv(TABLE_DIR / "Table1_Parameters_and_Sources.csv", index=False)
    pd.DataFrame([{"Scenario": k, **v.model_dump()} for k, v in scenarios.items()]).to_csv(
        TABLE_DIR / "Table2_Scenario_Configurations.csv", index=False
    )

    # --- Skenario Utama A, B, C1, C2, D ---
    all_yearly, all_reps = [], []
    for idx, (sc_key, sc_cfg) in enumerate(scenarios.items()):
        print(f"  [Main] Menjalankan {sc_key} ({n_reps} replikasi x {horizon} tahun)...", flush=True)
        y_df, r_df = run_scenario_montecarlo(
            sc_key, sc_cfg, demo_params, horizon_years=horizon, n_replicates=n_reps, seed=seed + idx * 1000
        )
        all_yearly.append(y_df)
        all_reps.append(r_df)

    yearly_full = pd.concat(all_yearly, ignore_index=True)
    reps_full = pd.concat(all_reps, ignore_index=True)
    yearly_summary = summarize_yearly(yearly_full)
    yearly_summary.to_csv(TABLE_DIR / "yearly_trajectories_summary.csv", index=False)

    def iqr_str(s):
        return f"{np.percentile(s, 25):.1f} - {np.percentile(s, 75):.1f}"

    summary_rows = []
    for sc_key in scenarios:
        sub = reps_full[reps_full["scenario"] == sc_key]
        n_sub = len(sub)
        k_ext = int(sub["extinct"].sum())
        k_qext = int(sub["quasi_extinct"].sum())
        summary_rows.append({
            "Scenario": sc_key,
            "Mean_Final_N": round(float(sub["final_n"].mean()), 2),
            "Median_Final_N": round(float(sub["final_n"].median()), 1),
            "IQR_Final_N": iqr_str(sub["final_n"]),
            "P5_P95_Final_N": f"{np.percentile(sub['final_n'], 5):.1f} - {np.percentile(sub['final_n'], 95):.1f}",
            "Extinction_Prob_Pct": round(float(sub["extinct"].mean() * 100.0), 2),
            "Extinction_95CI_Pct": wilson_ci_pct(k_ext, n_sub),
            "Quasi_Extinction_Prob_Pct": round(float(sub["quasi_extinct"].mean() * 100.0), 2),
            "Quasi_Ext_95CI_Pct": wilson_ci_pct(k_qext, n_sub),
            "Mean_Min_N": round(float(sub["min_n"].mean()), 1),
            "Mean_Final_He": round(float(sub["final_he"].mean()), 4),
            "Mean_Final_Ho": round(float(sub["final_ho"].mean()), 4),
            "Mean_Final_AR": round(float(sub["final_ar"].mean()), 3),
            "Mean_Final_F_IBD": round(float(sub["final_f_ibd"].mean()), 4),
            "Pct_He_Retained": round(float(100.0 * sub["final_he"].mean() / demo_params.initial_h0), 1),
        })
    main_summary_df = pd.DataFrame(summary_rows)
    main_summary_df[
        [
            "Scenario",
            "Mean_Final_N",
            "Median_Final_N",
            "IQR_Final_N",
            "P5_P95_Final_N",
            "Extinction_Prob_Pct",
            "Extinction_95CI_Pct",
            "Quasi_Extinction_Prob_Pct",
            "Quasi_Ext_95CI_Pct",
            "Mean_Min_N",
        ]
    ].to_csv(TABLE_DIR / "Table3_Demographic_Outcomes.csv", index=False)
    main_summary_df[
        ["Scenario", "Mean_Final_He", "Mean_Final_Ho", "Mean_Final_AR", "Mean_Final_F_IBD", "Pct_He_Retained"]
    ].to_csv(TABLE_DIR / "Table4_Genetic_Outcomes.csv", index=False)
    print("\nRingkasan Skenario Utama:")
    print(
        main_summary_df[
            [
                "Scenario",
                "Mean_Final_N",
                "Quasi_Extinction_Prob_Pct",
                "Extinction_Prob_Pct",
                "Mean_Final_He",
                "Mean_Final_F_IBD",
            ]
        ].to_string(index=False)
    )

    yrs = np.arange(0, horizon + 1)
    wf_df = pd.DataFrame({
        "year": yrs,
        "WF_Ne25": wright_fisher_curve(demo_params.initial_h0, 25.0, yrs),
        "WF_Ne50": wright_fisher_curve(demo_params.initial_h0, 50.0, yrs),
        "WF_Ne100": wright_fisher_curve(demo_params.initial_h0, 100.0, yrs),
    })
    wf_df.to_csv(TABLE_DIR / "wright_fisher_baselines.csv", index=False)

    # --- 5 Eksperimen Sensitivitas ---
    print("\n  [Exp 1] Demographic LHS + True PRCC sensitivity...", flush=True)
    exp1_samples, exp1_prcc = run_exp1_demographic_prcc(scenarios["A_Baseline_UKNP"], demo_params, seed)
    exp1_samples.to_csv(TABLE_DIR / "exp1_demographic_samples.csv", index=False)
    exp1_prcc.to_csv(TABLE_DIR / "Table5_Sensitivity_PRCC.csv", index=False)

    print("  [Exp 2] Genetic B x H0 sensitivity...", flush=True)
    exp2_df = run_exp2_genetics(scenarios["A_Baseline_UKNP"], demo_params, seed)
    exp2_df.to_csv(TABLE_DIR / "exp2_genetic_sensitivity.csv", index=False)

    print("  [Exp 3] Catastrophe spatial correlation sensitivity...", flush=True)
    exp3_df = run_exp3_catastrophe_correlation(scenarios["C_Metapop_Uncorrelated"], demo_params, seed)
    exp3_df.to_csv(TABLE_DIR / "exp3_catastrophe_correlation.csv", index=False)

    print("  [Exp 4] Carrying capacity K sensitivity...", flush=True)
    exp4_df = run_exp4_carrying_capacity(scenarios["A_Baseline_UKNP"], demo_params, seed)
    exp4_df.to_csv(TABLE_DIR / "exp4_carrying_capacity.csv", index=False)

    print("  [Exp 5] Combined strategy robustness matrix...", flush=True)
    exp5_df = run_exp5_combined_robustness(scenarios, demo_params, seed)
    exp5_df.to_csv(TABLE_DIR / "exp5_combined_robustness.csv", index=False)

    # --- Validasi Historis & Vortex 10 Bridge ---
    hist_df = run_historical_consistency_check(demo_params, seed)
    hist_df.to_csv(TABLE_DIR / "validation_historical_window.csv", index=False)

    export_vortex_spec_and_xml(scenarios, demo_params, VORTEX_DIR)
    table6 = build_cross_model_comparison_table(main_summary_df, demo_params, seed=seed)
    table6.to_csv(TABLE_DIR / "Table6_Python_vs_Vortex.csv", index=False)

    meta_run = {
        "random_seed": seed,
        "horizon_years": horizon,
        "n_replicates": n_reps,
        "min_interbirth_years_enforced": demo_params.min_interbirth_years,
        "lhs_samples_exp1": len(exp1_samples),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(meta_run, indent=2), encoding="utf-8")
    print("\nSelesai seluruh simulasi & tabel di:", TABLE_DIR)


if __name__ == "__main__":
    main()
