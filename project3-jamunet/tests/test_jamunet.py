"""Unit test komprehensif untuk JamuNet (deskriptor RDKit, fingerprint Tanimoto, PAINS/Brenk filter, & Scaffold Split)."""

import importlib
import sys
from pathlib import Path

import pandas as pd
from rdkit import DataStructs

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

dl_mod = importlib.import_module("02_druglikeness")
sim_mod = importlib.import_module("07_similarity_ml")
qsar_mod = importlib.import_module("08_qsar_3d_conformers")


def test_lipinski_ethanol_passes():
    row = pd.Series({"compound": "ethanol", "cid": 702, "plants": "test", "compound_class": "alcohol", "smiles": "CCO"})
    res = dl_mod.hitung_deskriptor(row)
    assert res["lipinski_pass"] is True
    assert res["lipinski_violations"] == 0
    assert 45.0 < res["mw"] < 47.0


def test_tanimoto_self_similarity_is_one():
    fp = sim_mod.smiles_to_fp("O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c12")
    assert DataStructs.TanimotoSimilarity(fp, fp) == 1.0


def test_pains_filter_flags_clean_vs_catechol():
    df = pd.DataFrame([
        {"compound": "caffeine", "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"},
        {"compound": "caffeic_acid", "smiles": "O=C(O)/C=C/c1ccc(O)c(O)c1"},
        {"compound": "naringenin", "smiles": "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c12"},
    ])
    res = qsar_mod.cek_pains_brenk(df).set_index("compound")
    assert res.loc["caffeine", "n_pains_alerts"] == 0
    assert res.loc["caffeic_acid", "n_pains_alerts"] >= 1
    assert res.loc["naringenin", "n_pains_alerts"] == 0
    assert res.loc["naringenin", "n_brenk_alerts"] == 0


def test_scaffold_split_guarantees_zero_scaffold_overlap():
    df = pd.DataFrame({
        "canon_smiles": [f"C{i}" for i in range(12)],
        "scaffold": ["S1"] * 5 + ["S2"] * 4 + ["S3"] * 2 + ["S4"],
        "pchembl": [6.5] * 12,
    })
    tr_idx, te_idx = qsar_mod.scaffold_split(df, test_frac=0.25)
    tr_scafs = set(df.iloc[tr_idx]["scaffold"])
    te_scafs = set(df.iloc[te_idx]["scaffold"])
    assert len(tr_scafs.intersection(te_scafs)) == 0
