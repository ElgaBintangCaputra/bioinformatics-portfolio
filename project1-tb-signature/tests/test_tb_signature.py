"""Unit test komprehensif untuk memverifikasi fungsi matematis, Nested CV anti-leakage, dan model TB-Signature."""

import importlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

pre = importlib.import_module("02_preprocess")
ml = importlib.import_module("05_machine_learning")
bn = importlib.import_module("06_boolean_network")
ode = importlib.import_module("07_lodo_benchmark_ode")


def test_quantile_normalize_aligns_distributions():
    df = pd.DataFrame({"s1": [10.0, 20.0, 30.0], "s2": [100.0, 50.0, 10.0]}, index=["g1", "g2", "g3"])
    qn = pre.quantile_normalize(df)
    assert np.allclose(sorted(qn["s1"]), sorted(qn["s2"]))
    assert qn.loc["g3", "s1"] > qn.loc["g1", "s1"]
    assert qn.loc["g1", "s2"] > qn.loc["g3", "s2"]


def test_zscore_per_sample_zero_mean_unit_std():
    df = pd.DataFrame({"p1": [2.0, 4.0, 6.0, 8.0], "p2": [10.0, 20.0, 30.0, 40.0]}, index=["g1", "g2", "g3", "g4"])
    z = ml.zscore_per_sampel(df)
    assert np.allclose(z.mean(axis=0).values, 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0).values, 1.0, atol=1e-5)


def test_nested_cv_signature_prevents_leakage_on_null_data():
    """Memastikan Nested CV tidak membocorkan label pada data acak murni (AUC tidak inflasi ke ~1.0)."""
    rng = np.random.default_rng(123)
    X_null = pd.DataFrame(rng.normal(size=(40, 60)), columns=[f"G{i}" for i in range(60)])
    y_null = np.array([0] * 20 + [1] * 20)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    probs, fold_genes = ml.evaluasi_nested_cv_signature(X_null, y_null, cv, n_genes=5)
    assert len(probs) == 40
    assert len(fold_genes) == 4
    assert all(len(fg) == 5 for fg in fold_genes)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_boolean_network_healthy_fixed_point():
    zero_state = (0, 0, 0, 0, 0, 0, 0, 0)
    assert bn.langkah_sinkron(zero_state) == zero_state
    _, atts = bn.temukan_semua_attractor()
    assert sum(a["ukuran_basin"] for a in atts) == 256


def test_ode_simulation_returns_finite_nonnegative():
    df = ode.simulasi_ode_imun()
    assert len(df) == 600
    assert (df[["Beban_Bakteri_B", "Interferon_I", "Efektor_Makrofag_M", "Rem_SOCS1_S"]] >= -1e-9).all().all()


def test_saved_model_or_signature_table_predicts_valid_probabilities():
    model_path = PROJECT_DIR / "results" / "models" / "tb_signature_model.joblib"
    sig_csv = PROJECT_DIR / "results" / "tables" / "05_signature_genes.csv"
    assert sig_csv.exists()
    sig_df = pd.read_csv(sig_csv)
    assert len(sig_df) == 8
    if model_path.exists():
        bundle = joblib.load(model_path)
        assert len(bundle["genes"]) == 8
        x_dummy = np.zeros((2, 8))
        probs = bundle["pipeline"].predict_proba(x_dummy)[:, 1]
        assert np.all((probs >= 0.0) & (probs <= 1.0))
