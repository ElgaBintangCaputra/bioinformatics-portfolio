"""
simulator.py — Mesin Simulasi Terintegrasi (Menghubungkan Engine 01, 02, dan 03).

Menjalankan R replikasi Monte Carlo selama T tahun untuk satu skenario konservasi
atau satu set parameter sensitivitas, dan mengembalikan deret waktu ringkasan
serta metrik akhir per replikasi.
"""

import numpy as np
import pandas as pd

from engines.demographic import DemographicParams, RhinoPopulationIBM
from engines.environment import draw_environmental_survival_modifier, sample_correlated_catastrophes
from models.parameters import ScenarioConfig


def run_scenario_montecarlo(
    scenario_key: str,
    cfg: ScenarioConfig,
    demo_params: DemographicParams,
    horizon_years: int = 100,
    n_replicates: int = 250,
    quasi_ext_thresh: int = 15,
    seed: int = 2026,
    record_yearly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan n_replicates simulasi stokastik.
    Mengembalikan:
      - yearly_df: data tahunan per replikasi (N_total, N_site1, N_site2, He, Ho, AR, F_ibd, cat1, cat2)
      - summary_reps: ringkasan per replikasi (extinct, quasi_extinct, min_N, final_N, final_He, final_F)
    """
    master_rng = np.random.default_rng(seed)
    yearly_records = []
    rep_records = []

    for rep in range(n_replicates):
        rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
        pop = RhinoPopulationIBM(rng, cfg.initial_n_site1, demo_params)

        # Catat t=0
        if record_yearly:
            g0 = pop.genetics_summary()
            yearly_records.append({
                "scenario": scenario_key,
                "rep": rep,
                "year": 0,
                "n_total": pop.total_n(),
                "n_site1": pop.count_site(1),
                "n_site2": pop.count_site(2),
                "he": g0["he"],
                "ho": g0["ho"],
                "allelic_richness": g0["allelic_richness"],
                "mean_f_ibd": g0["mean_f_ibd"],
                "cat_site1": False,
                "cat_site2": False,
            })

        min_n = pop.total_n()
        ever_quasi = pop.total_n() <= quasi_ext_thresh
        ever_ext = pop.total_n() == 0
        cat1_count = 0
        cat2_count = 0
        both_cat_count = 0

        for yr in range(1, horizon_years + 1):
            # Pembentukan populasi kedua jika dijadwalkan pada tahun ini
            if cfg.n_sites == 2 and cfg.founding_year == yr and cfg.founder_size > 0:
                pop.translocate_founders(cfg.founder_size, cfg.translocation_mortality)

            # Pertukaran migran berkala untuk menjaga konektivitas genetik
            if (
                cfg.n_sites == 2
                and cfg.migrant_interval is not None
                and cfg.founding_year is not None
                and yr > cfg.founding_year
                and (yr - cfg.founding_year) % cfg.migrant_interval == 0
            ):
                pop.move_managed_migrants(cfg.migrants_per_event)

            k_map = {1: cfg.k_site1}
            if cfg.n_sites == 2 and (cfg.founding_year is None or yr >= cfg.founding_year):
                k_map[2] = cfg.k_site2

            c1, c2 = sample_correlated_catastrophes(
                rng, cfg.catastrophe_prob, cfg.catastrophe_corr, cfg.n_sites
            )
            if c1:
                cat1_count += 1
            if c2:
                cat2_count += 1
            if c1 and c2:
                both_cat_count += 1

            mod1 = draw_environmental_survival_modifier(
                rng, demo_params.env_sd_survival, cfg.chronic_survival_drag, cfg.management_surv_boost
            )
            mod2 = draw_environmental_survival_modifier(
                rng, demo_params.env_sd_survival, cfg.chronic_survival_drag * 0.5, cfg.management_surv_boost
            )

            pop.step_year(
                k_by_site=k_map,
                surv_mod_by_site={1: mod1, 2: mod2},
                catastrophe_by_site={1: c1, 2: c2},
                catastrophe_severity=cfg.catastrophe_severity,
            )

            n_tot = pop.total_n()
            min_n = min(min_n, n_tot)
            if n_tot <= quasi_ext_thresh:
                ever_quasi = True
            if n_tot == 0:
                ever_ext = True

            if record_yearly:
                g = pop.genetics_summary()
                yearly_records.append({
                    "scenario": scenario_key,
                    "rep": rep,
                    "year": yr,
                    "n_total": n_tot,
                    "n_site1": pop.count_site(1),
                    "n_site2": pop.count_site(2),
                    "he": g["he"],
                    "ho": g["ho"],
                    "allelic_richness": g["allelic_richness"],
                    "mean_f_ibd": g["mean_f_ibd"],
                    "cat_site1": c1,
                    "cat_site2": c2,
                })

        if record_yearly:
            last = yearly_records[-1]
            fin_he, fin_ho, fin_ar, fin_f = last["he"], last["ho"], last["allelic_richness"], last["mean_f_ibd"]
        else:
            gf = pop.genetics_summary()
            fin_he, fin_ho, fin_ar, fin_f = gf["he"], gf["ho"], gf["allelic_richness"], gf["mean_f_ibd"]

        rep_records.append({
            "scenario": scenario_key,
            "rep": rep,
            "final_n": pop.total_n(),
            "final_n_site1": pop.count_site(1),
            "final_n_site2": pop.count_site(2),
            "min_n": min_n,
            "extinct": ever_ext,
            "quasi_extinct": ever_quasi,
            "final_he": fin_he,
            "final_ho": fin_ho,
            "final_ar": fin_ar,
            "final_f_ibd": fin_f,
            "cat1_events": cat1_count,
            "cat2_events": cat2_count,
            "simultaneous_cat_events": both_cat_count,
        })

    return pd.DataFrame(yearly_records), pd.DataFrame(rep_records)
