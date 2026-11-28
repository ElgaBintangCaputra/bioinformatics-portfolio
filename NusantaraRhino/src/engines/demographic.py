"""
demographic.py — Engine 01: Individual-Based Model (IBM) Demografi Badak Jawa.

Setiap individu direpresentasikan oleh struktur objek:
  I_i = (age, sex, breeding_status, state_geno, ibd_geno, f_ibd, site, cooldown_years)

Ciri biologis kunci Badak Jawa yang ditegakkan oleh mesin ini:
1. Masa kebuntingan ~15-16 bulan + pengasuhan anakan -> interval antar-kelahiran
   minimal >= 3 tahun (cooldown_years = 3 setelah melahirkan). TIDAK ada asumsi
   reproduksi tahunan yang tidak realistis!
2. Satu anak per kelahiran (litter size = 1), rasio seks lahir ~50:50.
3. Kesintasan bergantung umur (calf age 0-1, subadult 2-5, adult 6-32, senescent >32)
   dan terhubung dengan inbreeding depression anakan S_calf * exp(-B * F_i).
4. Daya dukung habitat (Carrying Capacity K) menekan rekrutmen & kesintasan saat N > K.
"""

from dataclasses import dataclass

import numpy as np

from engines.genetics import (
    compute_genetic_metrics,
    create_founder_genotypes,
    inherit_offspring_genotype,
)


@dataclass
class DemographicParams:
    female_maturity_age: int = 6
    male_maturity_age: int = 8
    max_reproductive_age: int = 32
    max_age: int = 40
    min_interbirth_years: int = 3
    sex_ratio_female: float = 0.50
    calf_survival: float = 0.88
    subadult_survival: float = 0.95
    adult_survival: float = 0.97
    senescent_survival: float = 0.82
    annual_conception_prob: float = 0.45
    lethal_equivalents_B: float = 1.57
    initial_h0: float = 0.55
    n_loci: int = 20
    n_alleles: int = 6
    env_sd_survival: float = 0.025


class RhinoPopulationIBM:
    """Kumpulan individu Badak Jawa berbasis vektor/daftar untuk satu replikasi Monte Carlo."""

    def __init__(
        self,
        rng: np.random.Generator,
        initial_n: int,
        params: DemographicParams,
    ):
        self.rng = rng
        self.params = params
        state_g, ibd_g = create_founder_genotypes(
            rng, initial_n, params.n_loci, params.initial_h0, params.n_alleles
        )
        # Inisialisasi struktur umur realistis (sebagian anakan, subadult, mayoritas dewasa)
        ages = rng.choice(
            np.arange(1, 28),
            size=initial_n,
            p=self._initial_age_probs(27),
        )
        sexes = rng.choice(["F", "M"], size=initial_n, p=[params.sex_ratio_female, 1.0 - params.sex_ratio_female])
        # Sebagian betina dewasa sedang dalam cooldown/bunting saat t=0
        cooldowns = np.where(
            (sexes == "F") & (ages >= params.female_maturity_age),
            rng.integers(0, params.min_interbirth_years, size=initial_n),
            0,
        )

        self.ages = ages.astype(np.int32)
        self.sexes = sexes
        self.sites = np.ones(initial_n, dtype=np.int32)  # 1 = Ujung Kulon, 2 = Populasi Kedua
        self.cooldowns = cooldowns.astype(np.int32)
        self.state_geno = state_g
        self.ibd_geno = ibd_g
        self.f_ibd = np.zeros(initial_n, dtype=np.float32)

    @staticmethod
    def _initial_age_probs(n_ages: int) -> np.ndarray:
        w = np.exp(-0.04 * np.arange(n_ages))
        return w / w.sum()

    def count_site(self, site: int) -> int:
        return int(np.sum(self.sites == site))

    def total_n(self) -> int:
        return int(len(self.ages))

    def step_year(
        self,
        k_by_site: dict[int, int],
        surv_mod_by_site: dict[int, float],
        catastrophe_by_site: dict[int, bool],
        catastrophe_severity: float,
    ) -> int:
        """Menjalankan 1 tahun siklus hidup: reproduksi -> kesintasan -> bencana -> penuaan."""
        if self.total_n() == 0:
            return 0

        p = self.params
        births_total = 0

        # --- 1. REPRODUKSI (per lokasi) ---
        new_ages, new_sexes, new_sites, new_cds, new_sg, new_ibd, new_f = [], [], [], [], [], [], []
        for site_id, k_site in k_by_site.items():
            if k_site <= 0:
                continue
            n_site = self.count_site(site_id)
            if n_site < 2:
                continue

            # Tekanan kepadatan ringan pada peluang pembuahan saat mendekati/melewati K
            density_factor = max(0.15, min(1.0, (1.15 * k_site - n_site) / max(1.0, 0.35 * k_site)))

            receptive_f_idx = np.where(
                (self.sites == site_id)
                & (self.sexes == "F")
                & (self.ages >= p.female_maturity_age)
                & (self.ages <= p.max_reproductive_age)
                & (self.cooldowns == 0)
            )[0]
            mature_m_idx = np.where(
                (self.sites == site_id)
                & (self.sexes == "M")
                & (self.ages >= p.male_maturity_age)
                & (self.ages <= p.max_reproductive_age)
            )[0]

            if len(receptive_f_idx) == 0 or len(mature_m_idx) == 0:
                continue

            conception_prob = p.annual_conception_prob * density_factor
            for f_i in receptive_f_idx:
                if self.rng.random() < conception_prob:
                    m_i = int(self.rng.choice(mature_m_idx))
                    c_sg, c_ibd, c_f = inherit_offspring_genotype(
                        self.rng,
                        self.state_geno[f_i],
                        self.ibd_geno[f_i],
                        self.state_geno[m_i],
                        self.ibd_geno[m_i],
                    )
                    # Betina memasuki masa kebuntingan + pengasuhan -> cooldown minimal 3 tahun
                    self.cooldowns[f_i] = p.min_interbirth_years
                    new_ages.append(0)
                    new_sexes.append("F" if self.rng.random() < p.sex_ratio_female else "M")
                    new_sites.append(site_id)
                    new_cds.append(0)
                    new_sg.append(c_sg)
                    new_ibd.append(c_ibd)
                    new_f.append(c_f)
                    births_total += 1

        if births_total > 0:
            self.ages = np.concatenate([self.ages, np.array(new_ages, dtype=np.int32)])
            self.sexes = np.concatenate([self.sexes, np.array(new_sexes)])
            self.sites = np.concatenate([self.sites, np.array(new_sites, dtype=np.int32)])
            self.cooldowns = np.concatenate([self.cooldowns, np.array(new_cds, dtype=np.int32)])
            self.state_geno = np.concatenate([self.state_geno, np.stack(new_sg)], axis=0)
            self.ibd_geno = np.concatenate([self.ibd_geno, np.stack(new_ibd)], axis=0)
            self.f_ibd = np.concatenate([self.f_ibd, np.array(new_f, dtype=np.float32)])

        # --- 2. KESINTASAN TAHUNAN (umur + kepadatan + inbreeding depression + lingkungan) ---
        n_now = self.total_n()
        n_s1 = self.count_site(1)
        n_s2 = self.count_site(2)
        k1 = max(1, k_by_site.get(1, 80))
        k2 = max(1, k_by_site.get(2, 70))

        # Fase anakan bergantung induk (2 tahun: usia 0 dan 1 tahun) mengalami depresi inbreeding juvenil
        base_s = np.where(
            self.ages <= 1,
            p.calf_survival * np.exp(-p.lethal_equivalents_B * self.f_ibd),
            np.where(
                self.ages < p.female_maturity_age,
                p.subadult_survival,
                np.where(self.ages <= p.max_reproductive_age, p.adult_survival, p.senescent_survival),
            ),
        )
        # Tekanan kepadatan K per site
        if n_s1 > k1:
            drag1 = min(0.25, 0.18 * (n_s1 - k1) / k1)
            base_s = np.where(self.sites == 1, base_s - drag1, base_s)
        if n_s2 > k2:
            drag2 = min(0.25, 0.18 * (n_s2 - k2) / k2)
            base_s = np.where(self.sites == 2, base_s - drag2, base_s)

        mods = np.where(self.sites == 1, surv_mod_by_site.get(1, 0.0), surv_mod_by_site.get(2, 0.0))
        s_prob = np.clip(base_s + mods, 0.05, 0.995)

        if catastrophe_by_site.get(1, False):
            s_prob = np.where(self.sites == 1, s_prob * (1.0 - catastrophe_severity), s_prob)
        if catastrophe_by_site.get(2, False):
            s_prob = np.where(self.sites == 2, s_prob * (1.0 - catastrophe_severity), s_prob)

        alive = (self.ages < p.max_age) & (self.rng.random(n_now) < s_prob)
        self._filter_alive(alive)

        # --- 3. PENUAAN & KURANGI COOLDOWN ---
        if self.total_n() > 0:
            self.ages += 1
            self.cooldowns = np.maximum(0, self.cooldowns - 1)

        return births_total

    def _filter_alive(self, mask: np.ndarray) -> None:
        self.ages = self.ages[mask]
        self.sexes = self.sexes[mask]
        self.sites = self.sites[mask]
        self.cooldowns = self.cooldowns[mask]
        self.state_geno = self.state_geno[mask]
        self.ibd_geno = self.ibd_geno[mask]
        self.f_ibd = self.f_ibd[mask]

    def translocate_founders(self, founder_size: int, mortality_risk: float) -> int:
        """Memindahkan individu subadult/adult sehat dari site 1 ke site 2 (Skenario C)."""
        if founder_size <= 0:
            return 0
        cand_idx = np.where(
            (self.sites == 1) & (self.ages >= 4) & (self.ages <= 22)
        )[0]
        if len(cand_idx) < founder_size:
            cand_idx = np.where((self.sites == 1) & (self.ages >= 2))[0]
        take_n = min(founder_size, len(cand_idx))
        if take_n == 0:
            return 0

        # Usahakan seimbang betina & jantan
        f_cands = [i for i in cand_idx if self.sexes[i] == "F"]
        m_cands = [i for i in cand_idx if self.sexes[i] == "M"]
        self.rng.shuffle(f_cands)
        self.rng.shuffle(m_cands)
        half = take_n // 2
        chosen = f_cands[:half] + m_cands[: take_n - half]
        if len(chosen) < take_n:
            rest = [i for i in cand_idx if i not in set(chosen)]
            chosen.extend(rest[: take_n - len(chosen)])

        chosen_arr = np.array(chosen, dtype=np.int32)
        self.sites[chosen_arr] = 2

        # Risiko mortalitas akut saat penangkapan/translokasi (IUCN 2025)
        if mortality_risk > 0:
            alive = np.ones(self.total_n(), dtype=bool)
            for idx in chosen_arr:
                if self.rng.random() < mortality_risk:
                    alive[idx] = False
            self._filter_alive(alive)

        return self.count_site(2)

    def move_managed_migrants(self, n_migrants: int) -> None:
        """Pertukaran genetik terkelola berkala (site 1 <-> site 2) untuk mencegah isolasi genetik."""
        if n_migrants <= 0 or self.count_site(1) < 10 or self.count_site(2) < 4:
            return
        # Pindahkan 1 individu dari site 1->2 dan 1 dari site 2->1 (jika memungkinkan)
        c1 = np.where((self.sites == 1) & (self.ages >= 4) & (self.ages <= 18))[0]
        c2 = np.where((self.sites == 2) & (self.ages >= 4) & (self.ages <= 18))[0]
        if len(c1) > 0:
            self.sites[int(self.rng.choice(c1))] = 2
        if n_migrants >= 2 and len(c2) > 0:
            self.sites[int(self.rng.choice(c2))] = 1

    def genetics_summary(self, site: int | None = None) -> dict[str, float]:
        if site is None:
            return compute_genetic_metrics(self.state_geno, self.ibd_geno)
        mask = self.sites == site
        return compute_genetic_metrics(self.state_geno[mask], self.ibd_geno[mask])
