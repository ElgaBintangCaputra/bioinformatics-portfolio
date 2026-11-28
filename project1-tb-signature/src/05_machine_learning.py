"""
05_machine_learning.py
======================
Langkah 5: Klasifikasi Machine Learning, Nested CV Anti-Leakage, Seleksi 8-Gen Signature,
Kalibrasi Single-Sample vs Kohort, & Validasi Eksternal Lintas-Platform (GSE37250, n=537).

Konsep kunci metodologi ketat (Peer-Review Grade):
1. ANTI-DATA-LEAKAGE & NESTED CROSS-VALIDATION:
   - Untuk model full-feature, StandardScaler + SelectKBest dibungkus di dalam Pipeline CV.
   - Untuk panel ringkas 8-Gen Signature, kami mengevaluasi **Nested 5x4-Fold CV** di mana
     pemilihan 8 gen via L1 Lasso dilakukan MURNI di dalam outer training fold pada setiap
     lipatan, sehingga estimasi performa internal bebas dari post-selection bias!
2. DUA MODE KALIBRASI LINTAS-PLATFORM (GPL6947 -> GPL10558):
   - Mode A (Unsupervised Target-Cohort Calibration): z-score per sampel across 9.459 gen,
     diikuti standardisasi per gen secara unsupervised di kohort target (tanpa menyentuh label).
   - Mode B (True Single-Sample Frozen-Discovery Calibration): z-score per sampel across 9.459 gen,
     diikuti standardisasi menggunakan mean & std yang DIBEKUKAN dari Discovery Cohort,
     sehingga 1 pasien baru dapat diprediksi secara mandiri tanpa menunggu batch sampel lain!
3. PERMUTATION IMPORTANCE INTERNAL & OUT-OF-SAMPLE EKSTERNAL:
   Kontribusi masing-masing dari 8 gen diukur baik di Discovery maupun secara out-of-sample
   di kohort validasi eksternal GSE37250 (n=537).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"
MODEL_DIR = PROJECT_DIR / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def zscore_per_sampel(df_gene_by_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi internal tiap sampel (kolom) menjadi mean=0, std=1 across all genes.
    Menempatkan setiap pasien pada skala relatif internal yang seragam.
    """
    return df_gene_by_sample.sub(df_gene_by_sample.mean(axis=0), axis=1).div(
        df_gene_by_sample.std(axis=0) + 1e-8, axis=1
    )


def pilih_signature_kompak(
    X_df: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_genes: int = 8,
) -> list[str]:
    """
    Memilih tepat N gen terbaik menggunakan stabilitas koefisien L1 Logistic Regression
    di dalam lipatan CV pada data latih yang diberikan (tanpa menyentuh data uji!).
    """
    genes = X_df.columns.tolist()
    abs_coefs = np.zeros(len(genes))

    for train_idx, _ in cv.split(X_df.values, y):
        X_tr, y_tr = X_df.values[train_idx], y[train_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        k_cand = min(200, X_tr_s.shape[1])
        sel = SelectKBest(f_classif, k=k_cand).fit(X_tr_s, y_tr)
        mask = sel.get_support()
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.2, random_state=42, max_iter=2000
        )
        clf.fit(X_tr_s[:, mask], y_tr)
        abs_coefs[mask] += np.abs(clf.coef_[0])

    ranking = pd.Series(abs_coefs, index=genes).sort_values(ascending=False)
    return ranking.head(n_genes).index.tolist()


def evaluasi_nested_cv_signature(
    X_raw_df: pd.DataFrame,
    y: np.ndarray,
    outer_cv: StratifiedKFold,
    n_genes: int = 8,
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Nested Cross-Validation (5-Fold Outer x 4-Fold Inner):
    Pada setiap outer fold, pemilihan 8 gen dilakukan MURNI pada outer training set,
    lalu model dilatih hanya pada 8 gen tersebut di outer training set dan diuji pada outer test set.
    Sepenuhnya bebas dari post-selection feature leakage!
    """
    oof_probs = np.zeros(len(y), dtype=float)
    fold_genes = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw_df.values, y)):
        X_tr_df = X_raw_df.iloc[train_idx]
        y_tr = y[train_idx]
        X_te_df = X_raw_df.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + fold_idx)
        selected = pilih_signature_kompak(X_tr_df, y_tr, inner_cv, n_genes=n_genes)
        fold_genes.append(selected)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, random_state=42, max_iter=2000)),
        ])
        pipe.fit(X_tr_df[selected].values, y_tr)
        oof_probs[test_idx] = pipe.predict_proba(X_te_df[selected].values)[:, 1]

    return oof_probs, fold_genes


def evaluasi_model_cv(
    X_raw_df: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, dict]:
    """Bandingkan 3 algoritma full-feature + Nested CV 8-Gen Signature dengan Pipeline anti-leakage."""
    X = X_raw_df.values
    models = {
        "LogisticRegression (L1)": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=150)),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.25, random_state=42, max_iter=2000)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=150)),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=150)),
            ("clf", SVC(kernel="linear", C=0.05, probability=True, random_state=42)),
        ]),
    }

    rows = []
    oof_probs = {}
    for name, pipe in models.items():
        probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[name] = probs
        rows.append({
            "Model": name,
            "CV_AUC": round(float(roc_auc_score(y, probs)), 4),
            "CV_Accuracy": round(float(accuracy_score(y, preds)), 4),
            "CV_F1": round(float(f1_score(y, preds)), 4),
        })

    # Tambahkan Nested CV 8-Gen Signature (100% bebas post-selection leakage)
    nested_probs, _ = evaluasi_nested_cv_signature(X_raw_df, y, cv, n_genes=8)
    nested_preds = (nested_probs >= 0.5).astype(int)
    oof_probs["8-Gen Signature (Nested CV)"] = nested_probs
    rows.append({
        "Model": "8-Gen Signature (Nested 5x4-Fold CV)",
        "CV_AUC": round(float(roc_auc_score(y, nested_probs)), 4),
        "CV_Accuracy": round(float(accuracy_score(y, nested_preds)), 4),
        "CV_F1": round(float(f1_score(y, nested_preds)), 4),
    })

    return pd.DataFrame(rows).sort_values("CV_AUC", ascending=False), oof_probs


def main():
    print("== Langkah 5: Machine Learning, Nested CV, & Validasi Eksternal ==")
    expr_disc = pd.read_parquet(PROC_DIR / "discovery_expr.parquet")
    meta_disc = pd.read_csv(PROC_DIR / "discovery_meta.csv", index_col=0)
    expr_val = pd.read_parquet(PROC_DIR / "validation_expr.parquet")
    meta_val = pd.read_csv(PROC_DIR / "validation_meta.csv", index_col=0)

    # 1. Normalisasi internal per sampel (single-sample z-score across 9.459 gen)
    X_disc_raw = zscore_per_sampel(expr_disc).T
    X_val_raw = zscore_per_sampel(expr_val).T

    # Mode A: Kalibrasi baseline per gen secara unsupervised di dalam masing-masing kohort
    disc_mean = X_disc_raw.mean(axis=0)
    disc_std = X_disc_raw.std(axis=0) + 1e-8
    X_disc_df = (X_disc_raw - disc_mean) / disc_std
    X_val_cohort_cal = (X_val_raw - X_val_raw.mean(axis=0)) / (X_val_raw.std(axis=0) + 1e-8)

    # Mode B: True Single-Sample Frozen-Discovery Calibration (tanpa mengintip sampel lain di kohort uji!)
    X_val_single_sample = (X_val_raw - disc_mean) / disc_std

    y_disc = meta_disc["y_tb"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("-> Membandingkan model full-feature & Nested CV 8-Gen Signature (Anti-Leakage)...")
    cv_df, oof_probs = evaluasi_model_cv(X_disc_raw, y_disc, cv)
    cv_df.to_csv(TABLE_DIR / "05_cv_comparison.csv", index=False)
    print(cv_df.to_string(index=False))

    print("-> Memilih konsensus 8-Gen TB Signature kompak dari Discovery Cohort...")
    sig_genes = pilih_signature_kompak(X_disc_raw, y_disc, cv, n_genes=8)
    print("   8 Gen Signature:", sig_genes)

    # Evaluasi 5-fold CV pada panel tetap 8 gen & latih model final pada seluruh Discovery Cohort
    sig_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, random_state=42, max_iter=2000)),
    ])
    sig_oof = cross_val_predict(
        sig_pipe, X_disc_df[sig_genes].values, y_disc, cv=cv, method="predict_proba"
    )[:, 1]
    sig_cv_auc = float(roc_auc_score(y_disc, sig_oof))
    sig_cv_acc = float(accuracy_score(y_disc, (sig_oof >= 0.5).astype(int)))
    nested_auc = float(roc_auc_score(y_disc, oof_probs["8-Gen Signature (Nested CV)"]))
    print(
        f"   Performa 8-Gen Signature: Nested CV AUC = {nested_auc:.4f} | "
        f"Final Panel CV AUC = {sig_cv_auc:.4f} (Acc = {sig_cv_acc:.4f})"
    )

    # Fit final model pada seluruh Discovery Cohort
    sig_pipe.fit(X_disc_df[sig_genes].values, y_disc)

    # Hitung Permutation Importance baik di Discovery maupun secara Out-of-Sample di GSE37250!
    de = pd.read_csv(TABLE_DIR / "03_differential_expression.csv").set_index("gene")
    coefs = sig_pipe.named_steps["clf"].coef_[0]
    intercept = float(sig_pipe.named_steps["clf"].intercept_[0])
    scaler_step = sig_pipe.named_steps["scaler"]

    perm_disc = permutation_importance(
        sig_pipe, X_disc_df[sig_genes].values, y_disc, n_repeats=30, random_state=42, scoring="roc_auc"
    )
    perm_ext = permutation_importance(
        sig_pipe,
        X_val_cohort_cal[sig_genes].values,
        meta_val["y_tb"].values,
        n_repeats=30,
        random_state=42,
        scoring="roc_auc",
    )

    sig_table = pd.DataFrame({
        "gene": sig_genes,
        "coef_logreg": coefs,
        "intercept_logreg": intercept,
        "scaler_mean": scaler_step.mean_,
        "scaler_scale": scaler_step.scale_,
        "frozen_disc_raw_mean": [float(disc_mean[g]) for g in sig_genes],
        "frozen_disc_raw_std": [float(disc_std[g]) for g in sig_genes],
        "perm_importance_auc": perm_disc.importances_mean,
        "perm_importance_ext_auc": perm_ext.importances_mean,
        "log2FC_discovery": [de.loc[g, "log2FC"] if g in de.index else np.nan for g in sig_genes],
        "fdr_discovery": [de.loc[g, "fdr"] if g in de.index else np.nan for g in sig_genes],
    }).sort_values("perm_importance_auc", ascending=False)
    sig_table.to_csv(TABLE_DIR / "05_signature_genes.csv", index=False)

    # --- VALIDASI EKSTERNAL pada GSE37250 (n=537) ---
    val_probs_cohort = sig_pipe.predict_proba(X_val_cohort_cal[sig_genes].values)[:, 1]
    val_probs_single = sig_pipe.predict_proba(X_val_single_sample[sig_genes].values)[:, 1]

    meta_val = meta_val.copy()
    meta_val["prob_tb"] = val_probs_cohort
    meta_val["prob_tb_single_sample"] = val_probs_single
    meta_val["pred_tb"] = (val_probs_cohort >= 0.5).astype(int)
    meta_val.to_csv(TABLE_DIR / "05_validation_predictions.csv")

    subsets = {
        "Active_TB vs Latent_TB (Semua)": meta_val["group"].isin(["Active_TB", "Latent_TB"]),
        "Active_TB vs Other_Disease (Semua)": meta_val["group"].isin(["Active_TB", "Other_Disease"]),
        "Active_TB vs Semua Non-TB (LTBI + Other)": pd.Series(True, index=meta_val.index),
        "Subgrup HIV- (TB vs All Non-TB)": meta_val["hiv_status"] == "HIV-",
        "Subgrup HIV+ (TB vs All Non-TB)": meta_val["hiv_status"] == "HIV+",
    }

    ext_rows = []
    roc_curves = {}
    for label, mask in subsets.items():
        sub = meta_val.loc[mask]
        yt = sub["y_tb"].values
        yp = sub["prob_tb"].values
        yp_ss = sub["prob_tb_single_sample"].values
        auc = roc_auc_score(yt, yp)
        auc_ss = roc_auc_score(yt, yp_ss)
        acc = accuracy_score(yt, (yp >= 0.5).astype(int))
        f1 = f1_score(yt, (yp >= 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(yt, (yp >= 0.5).astype(int)).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ext_rows.append({
            "Evaluasi": label,
            "N_Sampel": len(sub),
            "N_Active_TB": int(yt.sum()),
            "AUC": round(float(auc), 4),
            "AUC_SingleSample_Frozen": round(float(auc_ss), 4),
            "Accuracy": round(float(acc), 4),
            "Sensitivity": round(float(sens), 4),
            "Specificity": round(float(spec), 4),
            "F1": round(float(f1), 4),
        })
        roc_curves[label] = roc_curve(yt, yp) + (auc,)

    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(TABLE_DIR / "05_external_validation.csv", index=False)
    print("\nHasil Validasi Eksternal (GSE37250 - Cohort-Calibrated vs True Single-Sample):")
    print(ext_df.to_string(index=False))

    # Simpan model + metadata signature untuk Streamlit app
    joblib.dump(
        {
            "pipeline": sig_pipe,
            "genes": sig_genes,
            "discovery_means": X_disc_df[sig_genes].mean().to_dict(),
            "discovery_stds": X_disc_df[sig_genes].std().to_dict(),
            "frozen_raw_means": disc_mean[sig_genes].to_dict(),
            "frozen_raw_stds": disc_std[sig_genes].to_dict(),
            "sig_cv_auc": sig_cv_auc,
            "nested_cv_auc": nested_auc,
        },
        MODEL_DIR / "tb_signature_model.joblib",
    )

    # --- Plot 1: ROC Curves (Internal CV + External Validation) ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for name, probs in oof_probs.items():
        fpr, tpr, _ = roc_curve(y_disc, probs)
        ls = "-." if "Nested" in name else "-"
        lw = 2.3 if "Nested" in name else 2.0
        axes[0].plot(fpr, tpr, lw=lw, ls=ls, label=f"{name} (AUC={roc_auc_score(y_disc, probs):.3f})")
    fpr_s, tpr_s, _ = roc_curve(y_disc, sig_oof)
    axes[0].plot(fpr_s, tpr_s, lw=2.5, ls="--", color="black", label=f"8-Gen Panel CV (AUC={sig_cv_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.5)
    axes[0].set_title("Discovery Cohort: 5-Fold & Nested CV (Anti-Leakage)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Spesifisitas)")
    axes[0].set_ylabel("True Positive Rate (Sensitivitas)")
    axes[0].legend(fontsize=8, loc="lower right")

    for label in [
        "Active_TB vs Latent_TB (Semua)",
        "Active_TB vs Other_Disease (Semua)",
        "Active_TB vs Semua Non-TB (LTBI + Other)",
    ]:
        fpr, tpr, _, auc = roc_curves[label]
        axes[1].plot(fpr, tpr, lw=2.2, label=f"{label} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k:", alpha=0.5)
    axes[1].set_title("External Validation GSE37250 (n=537, Lintas Platform)", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (1 - Spesifisitas)")
    axes[1].set_ylabel("True Positive Rate (Sensitivitas)")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_roc_curves.png", dpi=220)
    plt.close()

    # --- Plot 2: Kepentingan & Koefisien 8 Gen Signature (Discovery vs External GSE37250) ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sig_plot = sig_table.sort_values("perm_importance_auc")
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in sig_plot["coef_logreg"]]
    axes[0].barh(sig_plot["gene"], sig_plot["perm_importance_auc"], color=colors, alpha=0.9)
    axes[0].set_xlabel("Penurunan ROC-AUC (Discovery Cohort)")
    axes[0].set_title("Permutation Importance (Discovery, n=147)", fontweight="bold")

    sig_plot_ext = sig_table.sort_values("perm_importance_ext_auc")
    colors_ext = ["#d62728" if c > 0 else "#1f77b4" for c in sig_plot_ext["coef_logreg"]]
    axes[1].barh(sig_plot_ext["gene"], sig_plot_ext["perm_importance_ext_auc"], color=colors_ext, alpha=0.9)
    axes[1].set_xlabel("Penurunan ROC-AUC (External GSE37250)")
    axes[1].set_title("Out-of-Sample Importance (GSE37250, n=537)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_signature_importance.png", dpi=220)
    plt.close()
    print("Selesai! Model dan grafik ML tersimpan.")


if __name__ == "__main__":
    main()
