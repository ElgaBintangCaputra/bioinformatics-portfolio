"""
environment.py — Engine 03: Stokastisitas Lingkungan (Tekanan Kronis & Katastrofik Berkorelasi).

Membedakan dua jenis tekanan secara tegas:
1. Tekanan kronis: noise tahunan (sigma_env) + penurunan bertahap (chronic_survival_drag).
2. Tekanan katastrofik: kejadian langka mendadak (tsunami, vulkanik, wabah) pada dua lokasi
   menggunakan Gaussian Copula sehingga P(C1=1, C2=1) != P(C1=1) * P(C2=1) bila rho > 0.
"""

import numpy as np
from scipy.stats import norm


def sample_correlated_catastrophes(
    rng: np.random.Generator,
    prob: float,
    rho: float,
    n_sites: int,
) -> tuple[bool, bool]:
    """
    Mengambil kejadian bencana tahunan di 1 atau 2 lokasi dengan korelasi spasial rho in [0, 1].
    Jika rho = 0 -> independen; jika rho > 0 -> memakai kopula Gaussian bivariat.
    """
    if prob <= 0.0:
        return (False, False)
    if n_sites == 1:
        return (bool(rng.random() < prob), False)

    if rho <= 0.0:
        return (bool(rng.random() < prob), bool(rng.random() < prob))
    if rho >= 0.999:
        c = bool(rng.random() < prob)
        return (c, c)

    # Kopula Gaussian bivariat
    z1 = rng.normal()
    z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * rng.normal()
    thresh = norm.ppf(prob)
    return (bool(z1 < thresh), bool(z2 < thresh))


def draw_environmental_survival_modifier(
    rng: np.random.Generator,
    sigma_env: float,
    chronic_drag: float,
    management_boost: float,
) -> float:
    """
    Menghasilkan pengubah aditif tahunan untuk probabilitas bertahan hidup.
    Menggabungkan fluktuasi acak lingkungan tahunan, tekanan kronis, dan efek proteksi/manajemen.
    """
    noise = float(rng.normal(0.0, sigma_env))
    return noise - chronic_drag + management_boost
