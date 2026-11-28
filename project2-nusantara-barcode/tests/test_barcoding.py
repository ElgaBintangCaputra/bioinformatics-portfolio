"""Unit test komprehensif untuk NusantaraBarcode (jarak K2P, k-mer, statistik populasi, ORF mitokondria, & audit asimetris)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

k2p_mod = importlib.import_module("03_distance_barcode_gap")
div_mod = importlib.import_module("03b_diversity_selection")
ml_mod = importlib.import_module("05_ml_species_id")


def test_k2p_identical_sequences_is_zero():
    seq = "ATGCGATACGCTTGAATGC"
    assert k2p_mod.k2p_distance(seq, seq) == 0.0


def test_k2p_transition_vs_transversion_ordering():
    ref = "AAAAAAAAAA"
    ts = "GAAAAAAAAA"
    tv = "TAAAAAAAAA"
    assert k2p_mod.k2p_distance(ref, ts) > 0
    assert k2p_mod.k2p_distance(ref, tv) > 0


def test_kmer_vector_sums_to_one():
    idx = ml_mod.buat_kamus_kmer(4)
    vec = ml_mod.hitung_vektor_kmer("ATGCGATACGCTTGAATGC", 4, idx)
    assert len(vec) == 256
    assert np.isclose(vec.sum(), 1.0)


def test_population_stats_monomorphic():
    stats = div_mod.statistik_populasi(["ATGCATGC", "ATGCATGC", "ATGCATGC", "ATGCATGC"])
    assert stats["h"] == 1
    assert stats["S"] == 0
    assert stats["pi"] == 0.0


def test_vertebrate_mitochondrial_orf_verification():
    # Kodon TGA adalah Tryptophan (bukan stop!) di mitokondria vertebrata (Tabel 2), sedangkan TAA adalah stop
    seqs = ["ATG" + "TGA" * 5 + "CTG", "ATG" + "TGA" * 5 + "CTG"]
    best_frame, stops = div_mod.verifikasi_orf_mitokondria(seqs, table_id=2)
    assert best_frame == 0
    assert stops[0] == 0


def test_asymmetric_audit_does_not_accuse_clean_cluster(tmp_path):
    """
    Memastikan jika 1 spesimen spesies B salah label sehingga identik dengan klaster bersih spesies A,
    hanya spesimen B yang ditandai Exclude_From_Reference, sedangkan klaster A tidak ikut dituduh!
    """
    ids = [
        "A1|Lutjanus_a|Lutjanidae",
        "A2|Lutjanus_a|Lutjanidae",
        "A3|Lutjanus_a|Lutjanidae",
        "B1|Epinephelus_b|Epinephelidae",
        "B2|Epinephelus_b|Epinephelidae",
        "B_bad|Epinephelus_b|Epinephelidae",  # salah label, identik dengan A
    ]
    mat = np.array([
        [0.000, 0.002, 0.002, 0.220, 0.220, 0.000],
        [0.002, 0.000, 0.002, 0.220, 0.220, 0.002],
        [0.002, 0.002, 0.000, 0.220, 0.220, 0.002],
        [0.220, 0.220, 0.220, 0.000, 0.003, 0.220],
        [0.220, 0.220, 0.220, 0.003, 0.000, 0.220],
        [0.000, 0.002, 0.002, 0.220, 0.220, 0.000],
    ])
    csv_path = tmp_path / "dist.csv"
    pd.DataFrame(mat, index=ids, columns=ids).to_csv(csv_path)
    audit = div_mod.audit_misannotation(csv_path)
    flagged_ids = set(audit["seq_id"])
    assert "B_bad|Epinephelus_b|Epinephelidae" in flagged_ids
    assert "A1|Lutjanus_a|Lutjanidae" not in flagged_ids
