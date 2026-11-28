"""Unit test 5-layer validation untuk NusantaraRhino (Pydantic, biologi 15-16 bln, genetika, kopula, & True PRCC)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engines.demographic import DemographicParams, RhinoPopulationIBM
from engines.environment import sample_correlated_catastrophes
from engines.genetics import (
    compute_genetic_metrics,
    create_founder_genotypes,
    inherit_offspring_genotype,
    wright_fisher_curve,
)
from experiments.run_experiments import build_base_demo_params, compute_prcc, wilson_ci_pct
from models.parameters import load_scenarios, load_validated_parameters


def test_parameter_database_provenance_valid():
    params = load_validated_parameters(ROOT / "data" / "parameters.csv")
    assert len(params) >= 25
    assert params["gestation_months"].assumption_type == "Literature-derived"
    dp = build_base_demo_params(params)
    # Pastikan ceil(3.5) = 4 langkah diskrit tahunan ditegakkan secara eksplisit (3 tahun penuh blokir pasca-lahir)
    assert dp.min_interbirth_years == 4
    _, scenarios = load_scenarios(ROOT / "configs" / "scenarios.yaml")
    assert "A_Baseline_UKNP" in scenarios
    assert "D_Integrated_Managed_Metapop" in scenarios


def test_breeding_interval_enforces_multiyear_cooldown():
    """Memastikan tidak ada betina yang bisa melahirkan setiap tahun berturut-turut."""
    rng = np.random.default_rng(42)
    dp = DemographicParams(
        annual_conception_prob=1.0,
        min_interbirth_years=3,
        calf_survival=1.0,
        adult_survival=1.0,
        subadult_survival=1.0,
    )
    pop = RhinoPopulationIBM(rng, 20, dp)
    pop.sexes[:10] = "F"
    pop.sexes[10:] = "M"
    pop.ages[:] = 12
    pop.cooldowns[:] = 0
    b1 = pop.step_year({1: 100}, {1: 0.0}, {1: False}, 0.0)
    assert b1 == 10
    b2 = pop.step_year({1: 100}, {1: 0.0}, {1: False}, 0.0)
    assert b2 == 0


def test_mendelian_ibd_and_wright_fisher_curve():
    rng = np.random.default_rng(7)
    sg, ibd = create_founder_genotypes(rng, 30, 20, 0.55, 6)
    g0 = compute_genetic_metrics(sg, ibd)
    assert abs(g0["he"] - 0.55) < 0.12
    assert g0["mean_f_ibd"] == 0.0

    _, _, f_self = inherit_offspring_genotype(rng, sg[0], ibd[0], sg[0], ibd[0])
    assert 0.25 <= f_self <= 0.75

    wf = wright_fisher_curve(0.55, 50.0, np.array([0.0, 12.0]))
    assert np.isclose(wf[1], 0.55 * (1.0 - 1.0 / 100.0))


def test_correlated_catastrophe_copula_exceeds_independence():
    rng = np.random.default_rng(99)
    n_trials = 4000
    both_ind = sum(1 for _ in range(n_trials) if all(sample_correlated_catastrophes(rng, 0.05, 0.0, 2)))
    both_cor = sum(1 for _ in range(n_trials) if all(sample_correlated_catastrophes(rng, 0.05, 0.8, 2)))
    assert both_cor > both_ind * 3


def test_true_prcc_and_wilson_ci():
    rng = np.random.default_rng(2026)
    x1 = rng.uniform(0, 1, size=40)
    x2 = rng.uniform(0, 1, size=40)
    y = -3.0 * x1 + 0.8 * x2 + rng.normal(0, 0.05, size=40)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    prcc = compute_prcc(df, ["x1", "x2"], "y")
    assert prcc["x1"][0] < -0.9
    assert prcc["x2"][0] > 0.5
    ci_str = wilson_ci_pct(0, 300)
    assert "0.0 -" in ci_str
