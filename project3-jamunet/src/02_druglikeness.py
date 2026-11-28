"""
02_druglikeness.py
==================
Langkah 2: Menghitung deskriptor fisikokimia & aturan kelayakan obat (RDKit).

Apa yang dicek?
1. Aturan Lima Lipinski (Rule of Five) untuk kelayakan obat minum (oral):
   - Berat Molekul (MW) <= 500 Da
   - cLogP (lipofilisitas / kelarutan dalam lemak) <= 5
   - Donor Ikatan Hidrogen (HBD) <= 5
   - Akseptor Ikatan Hidrogen (HBA) <= 10
   (Umumnya lolos jika maksimal 1 pelanggaran)
2. Aturan Veber (ketersediaan hayati oral):
   - TPSA (Topological Polar Surface Area) <= 140 A^2
   - Ikatan dapat berputar (Rotatable Bonds) <= 10
3. QED (Quantitative Estimate of Drug-likeness): skor komposit 0..1 (makin dekat 1 makin mirip obat).
4. Menggambar grid struktur 2D seluruh senyawa jamu dengan RDKit.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED

from common import FIG_DIR, PLANT_COLORS, TABLE_DIR


def hitung_deskriptor(row: pd.Series) -> dict | None:
    mol = Chem.MolFromSmiles(str(row["smiles"]))
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    qed_val = QED.qed(mol)

    lipinski_viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    veber_ok = (tpsa <= 140) and (rot <= 10)
    return {
        "compound": row["compound"],
        "cid": row["cid"],
        "plants": row["plants"],
        "compound_class": row["compound_class"],
        "mw": round(mw, 2),
        "clogp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": int(hbd),
        "hba": int(hba),
        "rot_bonds": int(rot),
        "qed": round(qed_val, 3),
        "lipinski_violations": lipinski_viol,
        "lipinski_pass": lipinski_viol <= 1,
        "veber_pass": bool(veber_ok),
    }


def main():
    print("== Langkah 2: RDKit Drug-likeness & Struktur 2D ==")
    comp = pd.read_csv(TABLE_DIR / "01_compounds_pubchem.csv")
    rows = [hitung_deskriptor(r) for _, r in comp.iterrows()]
    df = pd.DataFrame([r for r in rows if r is not None]).sort_values("qed", ascending=False)
    df.to_csv(TABLE_DIR / "02_druglikeness.csv", index=False)

    n_lip = int(df["lipinski_pass"].sum())
    print(f"  Total senyawa dianalisis: {len(df)} | Lolos Lipinski (<=1 pelanggaran): {n_lip}/{len(df)}")
    print("  Top 8 QED tertinggi:")
    print(df[["compound", "plants", "mw", "clogp", "tpsa", "qed", "lipinski_pass"]].head(8).to_string(index=False))

    # 1. Gambar Grid Struktur 2D semua senyawa
    mols, legends = [], []
    for _, r in df.iterrows():
        smi = comp.loc[comp["compound"] == r["compound"], "smiles"].iloc[0]
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            tag = "OK" if r["lipinski_pass"] else f"V{r['lipinski_violations']}"
            legends.append(f"{r['compound']}\n(QED={r['qed']:.2f}, {tag})")

    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280, 230), legends=legends)
    img.save(str(FIG_DIR / "02_compound_structures_grid.png"))

    # 2. Plot Overview Drug-likeness (QED + Ruang Lipinski MW vs cLogP)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    df_plot = df.sort_values("qed")
    primary_plant = df_plot["plants"].str.split(";").str[0].str.strip()
    bar_cols = [PLANT_COLORS.get(p, "#7f7f7f") for p in primary_plant]
    axes[0].barh(df_plot["compound"], df_plot["qed"], color=bar_cols, alpha=0.9)
    axes[0].axvline(0.5, ls="--", color="#333333", lw=1, label="QED = 0.5")
    axes[0].set_title("Skor QED (Drug-likeness) Senyawa Jamu", fontweight="bold")
    axes[0].set_xlabel("Skor QED (0 - 1)")
    axes[0].tick_params(axis="y", labelsize=8)
    for p, col in PLANT_COLORS.items():
        axes[0].barh([], [], color=col, label=p)
    axes[0].legend(fontsize=7.5, loc="lower right")

    sns.scatterplot(
        data=df, x="mw", y="clogp", hue="lipinski_pass", size="qed",
        sizes=(40, 220), palette={True: "#2ca02c", False: "#d62728"},
        ax=axes[1], alpha=0.88
    )
    axes[1].axvline(500, ls="--", color="gray", lw=1)
    axes[1].axhline(5, ls="--", color="gray", lw=1)
    axes[1].set_title("Ruang Fisikokimia Lipinski (MW vs cLogP)", fontweight="bold")
    axes[1].set_xlabel("Berat Molekul (g/mol)")
    axes[1].set_ylabel("cLogP (Lipofilisitas)")
    for _, r in df.head(6).iterrows():
        axes[1].annotate(r["compound"], (r["mw"], r["clogp"]), xytext=(4, 3),
                         textcoords="offset points", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_druglikeness_overview.png", dpi=220)
    plt.close()
    print("Selesai Langkah 2.")


if __name__ == "__main__":
    main()
