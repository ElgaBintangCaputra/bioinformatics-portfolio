"""
06_boolean_network.py
=====================
Langkah 6 (Bonus Systems Biology): Model Jaringan Boolean Respons Imun Terhadap TBC.

Apa itu Boolean Network?
Di bidang Systems Biology & Computational Biology, tidak semua hal harus berupa
statistik atau machine learning. Kita bisa memodelkan logika sebab-akibat antar-gen
sebagai gerbang logika biner:
  - 1 = Aktif / Terekspresi Tinggi (ON)
  - 0 = Tidak Aktif / Rendah (OFF)

CATATAN PENTING:
Ini adalah model edukasional / kualitatif berbasis literatur imunologi TBC
(jalur Interferon-gamma & respons makrofag) untuk mendemonstrasikan konsep
*attractor* (titik keseimbangan biologis) pada sistem imun inang — BUKAN model
kuantitatif klinis.

Node dalam jaringan (8 node):
1. Mtb_Load : Keberadaan bakteri Mycobacterium tuberculosis
2. TLR2_4   : Reseptor pengenal patogen (Toll-like receptor 2/4)
3. IFNG     : Sinyal Interferon-gamma (sitokin utama pelawan TBC)
4. STAT1    : Faktor transkripsi pengaktif gen interferon
5. IRF1     : Interferon Regulatory Factor 1
6. GBP1     : Guanylate Binding Protein 1 (gen efektor pembunuh bakteri intraseluler)
7. FCGR1A   : Reseptor Fc-gamma 1A (penanda kuat peradangan darah TBC)
8. SOCS1    : Suppressor of Cytokine Signaling 1 (rem alami supaya inflamasi tidak berlebihan)

Dengan 8 node biner, total ruang keadaan (state space) = 2^8 = 256 keadaan.
Kita bisa mensimulasikan SEMUA 256 keadaan awal untuk menemukan seluruh attractor-nya!
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"

NODES = ["Mtb_Load", "TLR2_4", "IFNG", "STAT1", "IRF1", "GBP1", "FCGR1A", "SOCS1"]


def langkah_sinkron(state: tuple[int, ...]) -> tuple[int, ...]:
    """
    Menghitung keadaan berikutnya t+1 dari keadaan t menggunakan aturan logika Boolean.
    Aturan disusun berdasarkan literatur jalur sinyal Interferon pada infeksi M. tuberculosis:
      - TLR2_4 aktif jika ada Mtb_Load.
      - IFNG aktif jika TLR2_4 aktif (sinyal awal infeksi) dan tidak ditekan total, atau dipertahankan.
      - STAT1 aktif oleh IFNG, tetapi dihambat oleh rem SOCS1.
      - IRF1 dan FCGR1A diaktifkan oleh STAT1 (atau langsung sinyal IFNG/TLR).
      - GBP1 (efektor antimikroba) diaktifkan oleh STAT1 AND IRF1.
      - SOCS1 (umpan balik negatif) diaktifkan belakangan oleh STAT1 yang tinggi.
      - Mtb_Load tetap ada kecuali jika efektor pembunuh (GBP1 & FCGR1A) sudah aktif.
    """
    mtb, tlr, ifng, stat1, irf1, gbp1, fcgr1a, socs1 = state

    next_tlr = 1 if mtb == 1 else 0
    next_ifng = 1 if (tlr == 1 or (ifng == 1 and mtb == 1)) else 0
    next_stat1 = 1 if (ifng == 1 and socs1 == 0) else 0
    next_irf1 = 1 if (stat1 == 1 or (ifng == 1 and tlr == 1)) else 0
    next_gbp1 = 1 if (stat1 == 1 and irf1 == 1) else 0
    next_fcgr1a = 1 if (stat1 == 1 or irf1 == 1) else 0
    next_socs1 = 1 if (stat1 == 1 and irf1 == 1) else 0
    # Jika bakteri ada & mekanisme bunuh GBP1 belum berhasil membersihkan, bakteri bertahan
    next_mtb = 1 if (mtb == 1 and not (gbp1 == 1 and socs1 == 0 and fcgr1a == 1)) else 0

    return (next_mtb, next_tlr, next_ifng, next_stat1, next_irf1, next_gbp1, next_fcgr1a, next_socs1)


def temukan_semua_attractor() -> tuple[pd.DataFrame, list[dict]]:
    """Menelusuri lintasan semua 2^8 = 256 state untuk mengelompokkan basin of attraction."""
    semua_states = list(itertools.product([0, 1], repeat=len(NODES)))
    attractors = {}
    records = []

    for st in semua_states:
        jejak = [st]
        seen = {st: 0}
        cur = st
        while True:
            nxt = langkah_sinkron(cur)
            if nxt in seen:
                cycle_start = seen[nxt]
                cycle = tuple(jejak[cycle_start:])
                # Kanonisasi siklus dengan rotasi terkecil agar unik
                rotations = [cycle[i:] + cycle[:i] for i in range(len(cycle))]
                canon = min(rotations)
                if canon not in attractors:
                    attractors[canon] = len(attractors) + 1
                att_id = attractors[canon]
                records.append({
                    "initial_state": "".join(map(str, st)),
                    "initial_mtb": st[0],
                    "attractor_id": att_id,
                    "steps_to_attractor": cycle_start,
                    "cycle_length": len(canon),
                })
                break
            seen[nxt] = len(jejak)
            jejak.append(nxt)
            cur = nxt

    df_states = pd.DataFrame(records)
    att_summary = []
    for canon, aid in attractors.items():
        basin_size = (df_states["attractor_id"] == aid).sum()
        mean_profile = np.mean(canon, axis=0)
        label = "Sehat / Bebas Infeksi (Steady State)" if mean_profile[0] == 0 else "Infeksi Aktif / Osilasi Imun-SOCS1"
        att_summary.append({
            "attractor_id": aid,
            "tipe": "Fixed Point" if len(canon) == 1 else f"Limit Cycle (periode {len(canon)})",
            "interpretasi": label,
            "ukuran_basin": int(basin_size),
            "persen_state_space": round(100 * basin_size / len(semua_states), 1),
            **{node: round(float(mean_profile[i]), 2) for i, node in enumerate(NODES)},
        })
    return df_states, att_summary


def simulasi_lintasan(initial_state: tuple[int, ...], steps: int = 12) -> pd.DataFrame:
    """Simulasikan satu lintasan waktu tertentu dari t=0 sampai t=steps."""
    history = [initial_state]
    cur = initial_state
    for _ in range(steps):
        cur = langkah_sinkron(cur)
        history.append(cur)
    return pd.DataFrame(history, columns=NODES)


def main():
    print("== Langkah 6 (Bonus): Simulasi Boolean Network Imun TBC ==")
    df_states, att_summary = temukan_semua_attractor()
    att_df = pd.DataFrame(att_summary)
    att_df.to_csv(TABLE_DIR / "06_boolean_attractors.csv", index=False)
    print(att_df[["attractor_id", "tipe", "interpretasi", "ukuran_basin", "persen_state_space"]].to_string(index=False))

    # Simulasikan skenario infeksi baru: Mtb_Load = 1, semua gen imun awalnya 0
    traj_infeksi = simulasi_lintasan((1, 0, 0, 0, 0, 0, 0, 0), steps=12)
    traj_infeksi.to_csv(TABLE_DIR / "06_boolean_trajectory_infection.csv", index_label="step")

    # Visualisasi: (Kiri) Profil rata-rata Attractor, (Kanan) Lintasan waktu saat awal terinfeksi Mtb
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})

    att_mat = att_df.set_index("attractor_id")[NODES]
    att_mat.index = [f"Attractor {i}\n({r['persen_state_space']}%)" for i, r in att_df.iterrows()]
    sns.heatmap(
        att_mat, annot=True, cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.8,
        cbar_kws={"label": "Aktivitas Rata-rata"}, ax=axes[0]
    )
    axes[0].set_title("Attractor Jaringan Boolean (256 State Awal)", fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")

    sns.heatmap(
        traj_infeksi.T, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=1, linewidths=0.6,
        cbar=False, ax=axes[1]
    )
    axes[1].set_title("Simulasi Waktu: Awal Infeksi (Mtb=1, lainnya=0)", fontweight="bold")
    axes[1].set_xlabel("Langkah Waktu (t)")
    axes[1].set_ylabel("Komponen Jaringan")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_boolean_network.png", dpi=220)
    plt.close()
    print("Selesai! Grafik Boolean network tersimpan di:", FIG_DIR / "06_boolean_network.png")


if __name__ == "__main__":
    main()
