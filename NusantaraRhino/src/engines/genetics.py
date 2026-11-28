"""
genetics.py — Engine 02: Simulasi Genetika Populasi Diploid & Pedigree Inbreeding.

Dua lapisan analisis:
1. Lapisan empiris: merujuk pada bukti penurunan garis keturunan mitokondria historis (2020)
   sambil menjaga transparansi bahwa genom nuklir populasi hidup belum lengkap.
2. Lapisan simulatif: pewarisan Mendel pada L=20 lokus diploid tak tertaut per individu,
   melacak:
   - Heterozigositas teramati (Ho) dan ekspektasi Hardy-Weinberg (He)
   - Kekayaan alel rata-rata per lokus (Allelic Richness, A_R)
   - Koefisien inbreeding individu (F_i: proporsi lokus homozigot identik-karena-keturunan
     relatif terhadap alel unik pendiri, atau homozigositas terstandar)
   - Penurunan kesintasan anakan akibat inbreeding depression: S_calf * exp(-B * F_i)
   - Baseline matematis Wright-Fisher: H_t = H_0 * (1 - 1/(2*N_e))^t.
"""

import numpy as np


def wright_fisher_curve(h0: float, ne: float, years: np.ndarray, gen_time: float = 12.0) -> np.ndarray:
    """Baseline teoretis penurunan heterozigositas per tahun (skala generasi T = years / gen_time)."""
    if ne <= 0:
        return np.zeros_like(years, dtype=float)
    t_gen = years / gen_time
    return h0 * np.power(1.0 - 1.0 / (2.0 * ne), t_gen)


def create_founder_genotypes(
    rng: np.random.Generator,
    n_ind: int,
    n_loci: int,
    target_h0: float,
    n_alleles: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Membangkitkan genotipe diploid awal (shape: [n_ind, n_loci, 2]) dengan frekuensi alel
    Dirichlet yang dikalibrasi sehingga heterozigositas ekspektasi awal mendekati target_h0,
    serta ID alel pendiri unik untuk melacak Identity-By-Descent (IBD) sejati!
    """
    # Alel unik per individu untuk pelacakan IBD sejati:
    # Kita gunakan dua matriks:
    # 1. state_alleles: alel fungsional (0..n_alleles-1) untuk He, Ho, dan allelic richness.
    # 2. ibd_alleles: tiap alel awal pendiri diberi ID unik (0..2*n_ind-1) per lokus sehingga
    #    homozigositas IBD (ibd_0 == ibd_1) mengukur F pedigree eksak relatif terhadap pendiri!
    # Kalibrasi konsentrasi alel agar 1 - sum(p_k^2) ~ target_h0
    # Untuk distribusi seragam k alel, H = 1 - 1/k. Jika target_h0 = 0.55, pakai distribusi geometrik/Dirichlet.
    probs_list = []
    # Pilih alpha berdasarkan target_h0 secara numerik sederhana
    best_alpha = 1.0
    best_diff = 999.0
    for alpha in np.linspace(0.15, 3.5, 40):
        expected_h = 1.0 - (1.0 + alpha) / (n_alleles * alpha + 1.0)
        if abs(expected_h - target_h0) < best_diff:
            best_diff = abs(expected_h - target_h0)
            best_alpha = float(alpha)

    state_geno = np.zeros((n_ind, n_loci, 2), dtype=np.int32)
    for loc in range(n_loci):
        p = rng.dirichlet(np.full(n_alleles, best_alpha))
        state_geno[:, loc, 0] = rng.choice(n_alleles, size=n_ind, p=p)
        state_geno[:, loc, 1] = rng.choice(n_alleles, size=n_ind, p=p)

    # IBD alleles: setiap kromosom pendiri punya tag unik (0 .. 2*n_ind - 1) -> F_0 = 0 relatif pendiri
    ibd_geno = np.zeros((n_ind, n_loci, 2), dtype=np.int32)
    for loc in range(n_loci):
        tags = np.arange(2 * n_ind, dtype=np.int32).reshape(n_ind, 2)
        ibd_geno[:, loc, :] = tags

    return state_geno, ibd_geno


def inherit_offspring_genotype(
    rng: np.random.Generator,
    mother_state: np.ndarray,
    mother_ibd: np.ndarray,
    father_state: np.ndarray,
    father_ibd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pewarisan Mendel bebas pada L lokus dari ibu dan ayah.
    Mengembalikan (state_anak, ibd_anak, F_ibd_anak).
    """
    n_loci = mother_state.shape[0]
    m_pick = rng.integers(0, 2, size=n_loci)
    f_pick = rng.integers(0, 2, size=n_loci)
    idx = np.arange(n_loci)

    child_state = np.stack([mother_state[idx, m_pick], father_state[idx, f_pick]], axis=1)
    child_ibd = np.stack([mother_ibd[idx, m_pick], father_ibd[idx, f_pick]], axis=1)
    f_ibd = float(np.mean(child_ibd[:, 0] == child_ibd[:, 1]))
    return child_state, child_ibd, f_ibd


def compute_genetic_metrics(state_geno: np.ndarray, ibd_geno: np.ndarray) -> dict[str, float]:
    """Menghitung Ho, He, Allelic Richness (A_R), dan rata-rata inbreeding F_IBD populasi."""
    n = state_geno.shape[0]
    if n == 0:
        return {"ho": 0.0, "he": 0.0, "allelic_richness": 0.0, "mean_f_ibd": 1.0}

    n_loci = state_geno.shape[1]
    ho = float(np.mean(state_geno[:, :, 0] != state_geno[:, :, 1]))
    max_allele = int(state_geno.max()) + 1

    he_sum = 0.0
    ar_sum = 0
    corr = (2.0 * n) / (2.0 * n - 1.0) if n > 1 else 1.0
    total_alleles = 2.0 * n
    for loc in range(n_loci):
        counts = np.bincount(state_geno[:, loc, :].ravel(), minlength=max_allele)
        nz = counts[counts > 0]
        freqs = nz / total_alleles
        he_loc = (1.0 - float(np.sum(freqs * freqs))) * corr
        he_sum += min(1.0, max(0.0, he_loc))
        ar_sum += len(nz)

    mean_f = float(np.mean(ibd_geno[:, :, 0] == ibd_geno[:, :, 1]))
    return {
        "ho": ho,
        "he": he_sum / n_loci,
        "allelic_richness": ar_sum / n_loci,
        "mean_f_ibd": mean_f,
    }
