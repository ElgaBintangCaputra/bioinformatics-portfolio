"""
04_phylogeny.py
===============
Langkah 4: Rekonstruksi Pohon Filogenetik (Neighbor-Joining & Maximum-Likelihood IQ-TREE 2).

Dua metode utama dalam filogenetika:
1. Neighbor-Joining (NJ, berbasis jarak K2P): sangat cepat, menyusun pohon dengan
   mengelompokkan pasangan dengan jarak evolusi terdekat.
2. Maximum-Likelihood (IQ-TREE 2 + ModelFinder + 1000 Ultrafast Bootstrap):
   metode statistik berbasis model substitusi DNA terbaik (otomatis dipilih oleh MFP)
   yang menguji keandalan setiap cabang pohon dengan 1.000 ulangan bootstrap!

Rooting: Pohon di-root pada famili Scombridae (tuna/tongkol) sebagai outgroup karena
kakap (Lutjanidae) dan kerapu (Epinephelidae) sama-sama ikan karang demersal yang lebih
dekat kekerabatannya dibanding Scombridae (pelagis).
"""

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

from common import PROJECT_DIR, get_paths, load_config, setup_logger

FAM_COLORS = {
    "Lutjanidae": "#d62728",
    "Epinephelidae": "#2ca02c",
    "Scombridae": "#1f77b4",
}


def bangun_pohon_nj(dist_csv: Path, out_nwk: Path, outgroup_family: str) -> Phylo.BaseTree.Tree:
    """Bangun pohon Neighbor-Joining dari matriks jarak K2P."""
    df = pd.read_csv(dist_csv, index_col=0)
    names = df.index.tolist()
    # Format DistanceMatrix Biopython: segitiga bawah (termasuk diagonal 0)
    lower_tri = []
    for i in range(len(names)):
        lower_tri.append([float(df.iloc[i, j]) for j in range(i + 1)])
    dm = DistanceMatrix(names=names, matrix=lower_tri)
    tree = DistanceTreeConstructor().nj(dm)

    # Root pohon pada outgroup (ambil salah satu clade Scombridae)
    out_tips = [t for t in tree.get_terminals() if outgroup_family in t.name]
    if out_tips:
        try:
            mrca = tree.common_ancestor(out_tips)
            if mrca != tree.root:
                tree.root_with_outgroup(mrca)
        except Exception:
            tree.root_with_outgroup(out_tips[0])

    # Rapikan nama node internal (Inner1 -> kosong)
    for clade in tree.get_nonterminals():
        if clade.name and clade.name.startswith("Inner"):
            clade.name = None

    Phylo.write(tree, out_nwk, "newick")
    return tree


def jalankan_iqtree(aln_fasta: Path, out_prefix: Path, iqtree_exe: Path,
                    model: str, ufboot: int, outgroup_family: str, logger) -> Phylo.BaseTree.Tree | None:
    """Menjalankan IQ-TREE 2 untuk membangun pohon Maximum-Likelihood + Ultrafast Bootstrap."""
    if not iqtree_exe.exists():
        logger.warning(f"iqtree2.exe tidak ditemukan di {iqtree_exe}, melewati ML tree.")
        return None

    # Salin alignment ke folder trees sementara karena IQ-TREE menulis file di dekat input/prefix
    local_aln = out_prefix.parent / (out_prefix.name + "_input.fasta")
    shutil.copy2(aln_fasta, local_aln)

    cmd = [
        str(iqtree_exe),
        "-s", str(local_aln),
        "-m", model,
        "-B", str(ufboot),
        "-T", "AUTO",
        "-pre", str(out_prefix),
        "-redo",
        "-quiet",
    ]
    logger.info(f"Menjalankan IQ-TREE 2: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    treefile = Path(str(out_prefix) + ".treefile")
    if res.returncode != 0 or not treefile.exists():
        logger.warning(f"IQ-TREE gagal (code {res.returncode}): {res.stderr[:400]}")
        return None

    tree = Phylo.read(treefile, "newick")
    out_tips = [t for t in tree.get_terminals() if outgroup_family in t.name]
    if out_tips:
        try:
            mrca = tree.common_ancestor(out_tips)
            if mrca != tree.root:
                tree.root_with_outgroup(mrca)
        except Exception:
            tree.root_with_outgroup(out_tips[0])
    Phylo.write(tree, treefile, "newick")
    return tree


def buat_fasta_perwakilan_spesies(aln_fasta: Path, meta_csv: Path, out_fasta: Path) -> None:
    """Pilih 1 sekuens perwakilan terbaik per spesies untuk pohon ringkas tingkat spesies."""
    meta = pd.read_csv(meta_csv)
    # Prioritaskan sampel Indonesia dengan panjang terbesar
    meta["is_id"] = meta["country"].str.lower().eq("indonesia").astype(int)
    best = (meta.sort_values(["is_id", "length"], ascending=[False, False])
            .drop_duplicates("species"))
    best_acc = set(best["accession"])
    recs = [r for r in SeqIO.parse(aln_fasta, "fasta") if r.id.split("|")[0] in best_acc]
    # Rapikan label menjadi Species|Family
    for r in recs:
        parts = r.id.split("|")
        r.id = f"{parts[1]}|{parts[2]}"
        r.description = ""
    SeqIO.write(recs, out_fasta, "fasta")


def gambar_pohon(tree: Phylo.BaseTree.Tree, title: str, out_png: Path, show_bootstrap: bool = True) -> None:
    """Menggambar pohon filogenetik dengan pewarnaan label berdasarkan famili ikan."""
    n_tips = len(tree.get_terminals())
    fig_h = max(7.5, n_tips * 0.19)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    def label_func(clade):
        if clade.is_terminal():
            return clade.name.replace("_", " ")
        if show_bootstrap and clade.confidence is not None:
            return f"{int(clade.confidence)}" if clade.confidence >= 70 else ""
        return None

    def label_colors(name):
        for fam, col in FAM_COLORS.items():
            if fam in name:
                return col
        return "#333333"

    tree.ladderize()
    Phylo.draw(
        tree,
        label_func=label_func,
        label_colors=label_colors,
        do_show=False,
        axes=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Substitusi per situs (panjang cabang)")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legenda famili
    for fam, col in FAM_COLORS.items():
        ax.plot([], [], color=col, lw=4, label=fam)
    ax.legend(title="Famili", loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    logger.info("== Langkah 4: Rekonstruksi Pohon Filogenetik ==")

    trees_dir = PROJECT_DIR / "results" / "trees"
    aln_trim = paths["processed_dir"] / "coi_aligned_trimmed.fasta"
    meta_csv = paths["tables_dir"] / "02_aligned_metadata.csv"
    outgroup = cfg["phylogeny"]["outgroup_family"]

    # 1. Pohon Neighbor-Joining (semua sekuens)
    nj_tree = bangun_pohon_nj(
        paths["tables_dir"] / "03_k2p_distance_matrix.csv",
        trees_dir / "nj_full.nwk",
        outgroup,
    )
    gambar_pohon(
        nj_tree,
        "Pohon Filogenetik Neighbor-Joining (K2P) — Seluruh Sekuens COI",
        paths["figures_dir"] / "04_nj_full_tree.png",
        show_bootstrap=False,
    )

    # 2. Pohon Maximum-Likelihood IQ-TREE 2 (perwakilan 1 sekuens per spesies + seluruh sampel)
    iqtree_exe = PROJECT_DIR / "tools" / "iqtree-2.3.6-Windows" / "bin" / "iqtree2.exe"

    sp_fasta = paths["processed_dir"] / "coi_species_rep.fasta"
    buat_fasta_perwakilan_spesies(aln_trim, meta_csv, sp_fasta)

    ml_sp_tree = jalankan_iqtree(
        sp_fasta,
        trees_dir / "iqtree_species",
        iqtree_exe,
        cfg["phylogeny"]["iqtree_model"],
        cfg["phylogeny"]["iqtree_ufboot"],
        outgroup,
        logger,
    )
    if ml_sp_tree is not None:
        gambar_pohon(
            ml_sp_tree,
            "Pohon Filogenetik Maximum-Likelihood (IQ-TREE 2, UFBoot=1000) — Tingkat Spesies",
            paths["figures_dir"] / "04_species_tree_ml.png",
            show_bootstrap=True,
        )

    ml_full_tree = jalankan_iqtree(
        aln_trim,
        trees_dir / "iqtree_full",
        iqtree_exe,
        cfg["phylogeny"]["iqtree_model"],
        cfg["phylogeny"]["iqtree_ufboot"],
        outgroup,
        logger,
    )
    if ml_full_tree is not None:
        gambar_pohon(
            ml_full_tree,
            "Pohon Filogenetik Maximum-Likelihood (IQ-TREE 2, UFBoot=1000) — Seluruh Sampel COI",
            paths["figures_dir"] / "04_full_tree_ml.png",
            show_bootstrap=True,
        )

    # Catat model substitusi terbaik yang dipilih IQ-TREE ModelFinder
    log_iq = trees_dir / "iqtree_species.iqtree"
    if log_iq.exists():
        for line in log_iq.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Best-fit model according to BIC"):
                logger.info(f"IQ-TREE ModelFinder: {line}")
                break

    logger.info("Selesai Langkah 4.")


if __name__ == "__main__":
    main()
