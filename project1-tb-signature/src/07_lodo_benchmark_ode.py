"""
07_lodo_benchmark_ode.py
========================
Langkah 7 (Pendalaman Riset & Validasi Mekanistik):
1. Strict Leave-One-Dataset-Out (LODO) Validation antar kohort geografi (London vs Afrika Selatan):
   - Strict Leak-Free LODO: pemilihan 8 gen dilakukan ULANG murni dari 2 kohort latih
     tanpa menyentuh kohort uji (holdout dataset).
   - Fixed-8 Consensus LODO: evaluasi panel 8 gen konsensus Discovery lintas kohort.
2. Benchmark Head-to-Head di GSE37250 (n=537) melawan signature literatur utuh:
   - Sweeney 3-gene score utuh = (GBP5 + DUSP3) / 2 - KLF2 (Lancet Respir Med, 2016)
   - Single-gene GBP5 score
   - 8-Gen TB-Signature
3. Pemodelan Matematika Biologi Kontinu (Sistem ODE dengan scipy.integrate.solve_ivp):
   Memodelkan dinamika waktu beban bakteri Mtb B(t), sinyal IFN-gamma I(t),
   aktivasi makrofag M(t), dan umpan balik negatif SOCS1 S(t) dalam 3 rezim klinis:
   Sembuh/Bersih, TBC Laten (keseimbangan), dan TBC Aktif.
"""

import importlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"
MODEL_PATH = PROJECT_DIR / "results" / "models" / "tb_signature_model.joblib"


def zscore_per_sampel(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0) + 1e-8, axis=1)


def evaluasi_lodo(expr_disc: pd.DataFrame, meta_disc: pd.DataFrame, sig_genes: list[str]) -> pd.DataFrame:
    """
    Leave-One-Dataset-Out (LODO):
    1. Strict LODO (Bebas Kebocoran Seleksi Fitur): memilih 8 gen murni dari 2 kohort latih,
       melatih model pada 2 kohort tersebut, lalu menguji pada 1 kohort holdout.
    2. Fixed-8 Panel LODO: melatih model dengan 8 gen konsensus pada 2 kohort latih.
    """
    ml_mod = importlib.import_module("05_machine_learning")
    X_raw = zscore_per_sampel(expr_disc).T
    rows = []

    for idx, holdout in enumerate(sorted(meta_disc["dataset"].unique())):
        tr_mask = (meta_disc["dataset"] != holdout).values
        te_mask = (meta_disc["dataset"] == holdout).values

        X_tr_all = X_raw.loc[tr_mask]
        X_te_all = X_raw.loc[te_mask]
        y_tr = meta_disc.loc[tr_mask, "y_tb"].values
        y_te = meta_disc.loc[te_mask, "y_tb"].values

        # 1. Strict LODO: seleksi 8 gen HANYA pada 2 kohort latih (tr_mask)
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + idx)
        strict_genes = ml_mod.pilih_signature_kompak(X_tr_all, y_tr, inner_cv, n_genes=8)

        X_tr_strict = (X_tr_all[strict_genes] - X_tr_all[strict_genes].mean()) / (
            X_tr_all[strict_genes].std() + 1e-8
        )
        X_te_strict = (X_te_all[strict_genes] - X_te_all[strict_genes].mean()) / (
            X_te_all[strict_genes].std() + 1e-8
        )
        pipe_strict = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, random_state=42, max_iter=2000)),
        ])
        pipe_strict.fit(X_tr_strict.values, y_tr)
        probs_strict = pipe_strict.predict_proba(X_te_strict.values)[:, 1]

        # 2. Fixed-8 Panel LODO
        X_tr = X_tr_all[sig_genes]
        X_te = X_te_all[sig_genes]
        X_tr_s = (X_tr - X_tr.mean()) / (X_tr.std() + 1e-8)
        X_te_s = (X_te - X_te.mean()) / (X_te.std() + 1e-8)

        pipe_fixed = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, random_state=42, max_iter=2000)),
        ])
        pipe_fixed.fit(X_tr_s.values, y_tr)
        probs_fixed = pipe_fixed.predict_proba(X_te_s.values)[:, 1]

        rows.append({
            "Kohort_Uji": holdout,
            "N_Sampel": int(te_mask.sum()),
            "N_Active_TB": int(y_te.sum()),
            "ROC_AUC_Strict_LODO": round(float(roc_auc_score(y_te, probs_strict)), 4),
            "PR_AUC_Strict_LODO": round(float(average_precision_score(y_te, probs_strict)), 4),
            "ROC_AUC": round(float(roc_auc_score(y_te, probs_fixed)), 4),
            "PR_AUC": round(float(average_precision_score(y_te, probs_fixed)), 4),
            "Strict_Fold_Genes": ";".join(strict_genes),
        })
    return pd.DataFrame(rows)


def benchmark_sweeney3(
    expr_val: pd.DataFrame,
    meta_val: pd.DataFrame,
    model_bundle: dict,
    sweeney_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Membandingkan model 8-Gen TB-Signature kita terhadap:
    1. Skor penuh Sweeney 3-gene: (GBP5 + DUSP3) / 2 - KLF2
    2. Penanda tunggal GBP5
    di kohort eksternal GSE37250 (n=537).
    """
    X_val_raw = zscore_per_sampel(expr_val).T
    X_val_cal = (X_val_raw - X_val_raw.mean(axis=0)) / (X_val_raw.std(axis=0) + 1e-8)
    sig_genes = model_bundle["genes"]
    prob_ours = model_bundle["pipeline"].predict_proba(X_val_cal[sig_genes].values)[:, 1]

    # Gunakan matriks Sweeney pra-filter varians jika tersedia agar ketiga gen lengkap
    sw_source = sweeney_df if (sweeney_df is not None and not sweeney_df.empty) else expr_val
    sweeney_genes = ["GBP5", "DUSP3", "KLF2"]
    tersedia = [g for g in sweeney_genes if g in sw_source.index]

    if set(sweeney_genes).issubset(sw_source.index):
        # Rumus asli Sweeney et al. (Lancet Respir Med, 2016): (GBP5 + DUSP3)/2 - KLF2
        score_sweeney3 = (
            0.5 * (sw_source.loc["GBP5"].values + sw_source.loc["DUSP3"].values)
            - sw_source.loc["KLF2"].values
        )
    else:
        pos_g = [g for g in ["GBP5", "DUSP3"] if g in sw_source.index]
        score_sweeney3 = sw_source.loc[pos_g].mean(axis=0).values

    score_gbp5 = sw_source.loc["GBP5"].values if "GBP5" in sw_source.index else score_sweeney3

    tasks = {
        "Active TB vs Latent TB": meta_val["group"].isin(["Active_TB", "Latent_TB"]).values,
        "Active TB vs Other Disease": meta_val["group"].isin(["Active_TB", "Other_Disease"]).values,
        "Active TB vs Semua Non-TB": np.ones(len(meta_val), dtype=bool),
    }

    rows = []
    curves = {}
    y_all = meta_val["y_tb"].values
    for task_name, mask in tasks.items():
        yt = y_all[mask]
        po = prob_ours[mask]
        ps3 = score_sweeney3[mask]
        pg5 = score_gbp5[mask]

        auc_o = roc_auc_score(yt, po)
        auc_s3 = roc_auc_score(yt, ps3)
        auc_g5 = roc_auc_score(yt, pg5)
        ap_o = average_precision_score(yt, po)
        ap_s3 = average_precision_score(yt, ps3)
        ap_g5 = average_precision_score(yt, pg5)

        rows.append({
            "Tugas_Evaluasi": task_name,
            "Sweeney_Gen_Tersedia": "+".join(tersedia),
            "AUC_TB_Signature_8Gen": round(float(auc_o), 4),
            "AUC_Sweeney_3Gen": round(float(auc_s3), 4),
            "AUC_GBP5_Single": round(float(auc_g5), 4),
            "PR_AUC_8Gen": round(float(ap_o), 4),
            "PR_AUC_Sweeney": round(float(ap_s3), 4),
            "PR_AUC_GBP5_Single": round(float(ap_g5), 4),
        })
        curves[task_name] = (
            roc_curve(yt, po),
            auc_o,
            roc_curve(yt, ps3),
            auc_s3,
            roc_curve(yt, pg5),
            auc_g5,
        )
    return pd.DataFrame(rows), curves


def simulasi_ode_imun() -> pd.DataFrame:
    """
    Model Matematika Kontinu (ODE) dinamika bakteri M. tuberculosis vs Imun Inang:
      dB/dt = r*B*(1 - B/K) - k_kill * M * B / (1 + alpha*S)
      dI/dt = p_I * B / (0.4 + B) - d_I * I
      dM/dt = p_M * (I / (0.35 + I)) / (1 + beta*S) - d_M * M
      dS/dt = p_S * M * I - d_S * S
    """
    def rhs(t, y, r, k_kill):
        B, I, M, S = np.maximum(y, 0.0)
        K = 2.5
        dB = r * B * (1.0 - B / K) - k_kill * M * B / (1.0 + 0.8 * S)
        dI = 1.4 * B / (0.4 + B) - 0.65 * I
        dM = 1.6 * (I / (0.35 + I)) / (1.0 + 1.2 * S) - 0.55 * M
        dS = 0.75 * M * I - 0.85 * S
        return [dB, dI, dM, dS]

    regimes = [
        ("Imun Kuat (Eliminasi / Sembuh)", 0.35, 1.35),
        ("Keseimbangan Laten (TBC Laten)", 0.55, 0.90),
        ("Imun Terlampaui (TBC Aktif)", 0.85, 0.50),
    ]
    t_eval = np.linspace(0, 40, 200)
    records = []
    for label, r, k_kill in regimes:
        sol = solve_ivp(rhs, (0, 40), [0.3, 0.05, 0.05, 0.0], t_eval=t_eval, args=(r, k_kill))
        for idx, t in enumerate(sol.t):
            records.append({
                "Skenario": label,
                "Hari": float(t),
                "Beban_Bakteri_B": float(sol.y[0, idx]),
                "Interferon_I": float(sol.y[1, idx]),
                "Efektor_Makrofag_M": float(sol.y[2, idx]),
                "Rem_SOCS1_S": float(sol.y[3, idx]),
            })
    return pd.DataFrame(records)


def main():
    print("== Langkah 7: Strict LODO Cross-Cohort, Benchmark Sweeney-3 Utuh, & Model ODE ==")
    expr_disc = pd.read_parquet(PROC_DIR / "discovery_expr.parquet")
    meta_disc = pd.read_csv(PROC_DIR / "discovery_meta.csv", index_col=0)
    expr_val = pd.read_parquet(PROC_DIR / "validation_expr.parquet")
    meta_val = pd.read_csv(PROC_DIR / "validation_meta.csv", index_col=0)
    sw_val_path = PROC_DIR / "sweeney3_val_expr.parquet"
    sweeney_val = pd.read_parquet(sw_val_path) if sw_val_path.exists() else None
    bundle = joblib.load(MODEL_PATH)

    lodo_df = evaluasi_lodo(expr_disc, meta_disc, bundle["genes"])
    lodo_df.to_csv(TABLE_DIR / "07_lodo_validation.csv", index=False)
    print("\n1. Leave-One-Dataset-Out (LODO) Validation (Strict vs Fixed-8):")
    print(lodo_df.to_string(index=False))

    bench_df, curves = benchmark_sweeney3(expr_val, meta_val, bundle, sweeney_df=sweeney_val)
    bench_df.to_csv(TABLE_DIR / "07_benchmark_sweeney3.csv", index=False)
    print("\n2. Benchmark 8-Gen Signature vs Sweeney 3-Gen Utuh & GBP5 pada GSE37250 (n=537):")
    print(bench_df.to_string(index=False))

    ode_df = simulasi_ode_imun()
    ode_df.to_csv(TABLE_DIR / "07_ode_simulation.csv", index=False)

    # Plot gabungan: (Kiri) Benchmark ROC Active TB vs Non-TB, (Kanan) Dinamika ODE
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    (fpr_o, tpr_o, _), auc_o, (fpr_s3, tpr_s3, _), auc_s3, (fpr_g5, tpr_g5, _), auc_g5 = curves[
        "Active TB vs Semua Non-TB"
    ]
    axes[0].plot(fpr_o, tpr_o, lw=2.5, color="#d62728", label=f"TB-Signature 8-Gen (AUC={auc_o:.3f})")
    axes[0].plot(
        fpr_s3, tpr_s3, lw=2.2, ls="--", color="#1f77b4", label=f"Sweeney 3-Gen Utuh (AUC={auc_s3:.3f})"
    )
    axes[0].plot(
        fpr_g5, tpr_g5, lw=1.8, ls="-.", color="#2ca02c", label=f"GBP5 Single-Gene (AUC={auc_g5:.3f})"
    )
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.5)
    axes[0].set_title("Benchmark Validasi Eksternal GSE37250 (n=537)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=8.5)

    palette_ode = {
        "Imun Kuat (Eliminasi / Sembuh)": "#2ca02c",
        "Keseimbangan Laten (TBC Laten)": "#ff7f0e",
        "Imun Terlampaui (TBC Aktif)": "#d62728",
    }
    for scen, col in palette_ode.items():
        sub = ode_df[ode_df["Skenario"] == scen]
        axes[1].plot(sub["Hari"], sub["Beban_Bakteri_B"], lw=2.3, color=col, label=f"{scen} (Bakteri)")
        axes[1].plot(sub["Hari"], sub["Efektor_Makrofag_M"], lw=1.4, ls=":", color=col, alpha=0.85)
    axes[1].set_title("Model ODE Kontinu: Dinamika Bakteri (garis utuh) vs Imun (putus)", fontweight="bold")
    axes[1].set_xlabel("Waktu Simulasi (Hari)")
    axes[1].set_ylabel("Aktivitas / Beban Relatif")
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_benchmark_and_ode.png", dpi=220)
    plt.close()
    print("Selesai Langkah 7.")


if __name__ == "__main__":
    main()
