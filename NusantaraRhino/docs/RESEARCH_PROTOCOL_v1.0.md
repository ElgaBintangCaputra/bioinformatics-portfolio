# NusantaraRhino — Research Protocol v1.0

**Proposed Manuscript Title:**  
*Evaluating the Long-Term Viability of the Javan Rhinoceros (Rhinoceros sondaicus) under Alternative Conservation Strategies: An Integrated Stochastic Population and Genetic Modeling Approach*

**Status:** Protocol v1.0 (Locked prior to simulation runs)  
**Primary Computational Platform:** Python (IBM + Mendelian Genetics + Correlated Environmental Stochasticity)  
**Cross-Model Benchmark Framework:** Vortex 10 (SCTI) parameter-matched specification & structural comparison  

---

## 1. Research Background & Knowledge Gap

The Javan rhinoceros (*Rhinoceros sondaicus*) is among the rarest large mammals on Earth, restricted to a single wild population in Ujung Kulon National Park (UKNP), Banten, Indonesia. Because the entire global species resides in one coastal peninsula adjacent to the Sunda Strait (Krakatau volcanic arc and tsunami zone), the species faces acute vulnerability from both **demographic/genetic stochasticity** (small population size, drift, potential inbreeding) and **spatially correlated catastrophic risks**.

### What is already established?
1. **2015 Javan Rhino PHVA (Conservation Planning Specialist Group / CPSG):** Used Vortex to evaluate threats and second-population scenarios starting from ~58–61 individuals, establishing baseline demographic feasibility and highlighting the danger of single-site concentration.
2. **2020 Mitochondrial Lineage Extinction Study:** Analysis of 8 museum specimens alongside modern samples demonstrated that historical populations harbored greater maternal mitochondrial lineage diversity than surviving individuals (*Zoological Journal of the Linnean Society*, 2020), while also underscoring that modern nuclear genomic data remain incomplete.
3. **Reproductive Biology (IUCN SSC Asian Rhino Specialist Group):** Female Javan rhinos have an estimated gestation of ~15–16 months and produce a single calf with multi-year inter-birth intervals (typically 3–5 years), making unrealistic annual-breeding assumptions biologically invalid.
4. **Conservation Policy Context (IUCN 2025 Translocation Guidelines & BRIN 2026):** Establishing a second habitat (e.g., evaluating candidates such as Alas Purwo or alternative sites) remains an active research and feasibility question requiring ecological, social, management, and genetic connectivity assessment—not a settled operational decision.

### Research Gap & Novelty Contributions
Simply contrasting "Ujung Kulon alone vs second population" repeats the 2015 PHVA. This study targets three concrete contributions:
1. **Novelty 1 — Coupled Demographic–Genetic–Environmental IBM:** Explicit individual-based simulation linking 15–16 month gestation/interbirth constraints, multi-locus diploid Mendelian inheritance with inbreeding depression, and distinct chronic vs catastrophic environmental drivers.
2. **Novelty 2 — Spatially Correlated Catastrophe Modeling ($P(C_1, C_2) \neq P(C_1)P(C_2)$):** Testing whether geographic diversification still reduces global extinction risk when regional catastrophes (volcanic ashfall, tsunami, disease spillover) exhibit positive inter-site correlation $\rho \in [0, 0.9]$.
3. **Novelty 3 — Decision Robustness under Parameter Uncertainty:** Identifying conservation strategies that remain robust across broad uncertainty intervals for poorly measured parameters (nuclear heterozygosity $H_0$, effective population size $N_e$, lethal equivalents $B$, and carrying capacity $K$).

---

## 2. Research Questions & Hypotheses

- **H1 (Single-population vulnerability):** A single concentrated population in UKNP exhibits substantially higher 100-year quasi-extinction and global extinction risk under regional catastrophic hazards than diversified populations with low catastrophe correlation.
- **H2 (Management intervention):** Reducing chronic anthropogenic/habitat pressures and improving calf/adult survival & carrying capacity ($K$) measurably increases mean population trajectory and lowers extinction risk even in a single site.
- **H3 (Genetic sustainability):** Even when demographic numbers stabilize, small $N_e$ leads to steady erosion of expected heterozygosity ($H_e$) and rising pedigree/locus inbreeding ($F$), potentially depressing calf survival under high genetic load ($B$).
- **H4 (Metapopulation resilience — conditional):** Establishing a second population improves global persistence *only conditionally*: small un-supplemented founder groups suffer severe founder drift and inbreeding, requiring minimum founder sizes, low inter-site catastrophe correlation, and periodic genetic exchange (*one-migrant-per-generation* / managed translocation).

---

## 3. Provenance Rules & Epistemic Discipline

Every parameter in `data/parameters.csv` is tagged with an explicit `assumption_type`:
- `Observed`: directly observed empirical count/measurement.
- `Literature-derived`: extracted from peer-reviewed papers or CPSG/IUCN reports.
- `Estimated`: derived statistically from published ranges.
- `Assumed`: explicit modeling assumption tested via sensitivity analysis.
- `Scenario-defined`: experimental policy/intervention knob.

> **Epistemic Guardrail:** Unverified summary figures (such as unstandardized inbreeding percentages like `58.5%` without verified statistical denominator/ROH method) are **excluded** as direct point inputs and instead bracketed via transparent sensitivity ranges ($H_0$, $B$).

---

## 4. Mathematical & Computational Architecture

### Engine 01 — Demographic IBM
Each individual $i$ is represented by state tuple:
$$I_i = (a_i, s_i, b_i, g_i, \text{loc}_i, c_i)$$
where $a_i$ is age (years), $s_i \in \{\text{F}, \text{M}\}$ is sex, $b_i$ is reproductive/gestation status, $g_i$ is diploid genotype array across $L = 20$ unlinked loci, $\text{loc}_i \in \{1, 2\}$ is site, and $c_i$ is post-birth/gestation cooldown years ($\ge 3$ years to honor 15–16 month gestation + lactation).

### Engine 02 — Population Genetics
- **Theoretical Wright–Fisher baseline:** $H_t = H_0 \left(1 - \frac{1}{2N_e}\right)^t$
- **Individual-based Mendelian engine:** Offspring inherit one random allele per locus from each parent. Tracks $H_o$, $H_e$, allelic richness $A_R$, pedigree/identity-by-descent inbreeding $F_i$, and realized $N_e$.
- **Inbreeding depression:** Calf first-year survival is modulated by $S_{\text{calf}, i} = S_{\text{calf}, 0} \cdot \exp(-B \cdot F_i)$, where $B$ (haploid lethal equivalents) is varied across sensitivity experiments.

### Engine 03 — Environmental & Catastrophic Stochasticity
- **Chronic pressure:** Gradual annual survival / $K$ modifier drawn from Beta/Normal environmental noise ($\sigma_{\text{env}}$) plus optional chronic habitat degradation trend.
- **Catastrophic pressure:** Correlated bivariate Bernoulli events $(C_1, C_2)$ generated via Gaussian copula with correlation $\rho \in \{0.0, 0.3, 0.6, 0.9\}$, ensuring $P(C_1=1, C_2=1) > P(C_1=1)P(C_2=1)$ whenever $\rho > 0$.

---

## 5. Experimental Design & Validation Strategy

- **Core Scenarios (100-year horizon, 250 Monte Carlo runs per scenario):**
  - **Scenario A:** Single population (Ujung Kulon baseline, $N_0 \approx 76$ or 2015/modern sensitivity anchor, $K = 80$).
  - **Scenario B:** Ujung Kulon intensive management (chronic pressure alleviated, $K$ expanded to $100$, survival enhanced).
  - **Scenario C:** Metapopulation (Ujung Kulon + Second Population founded at $t=5$ with $N_{\text{founder}} = 14$, plus managed genetic exchange every 10 years, tested across catastrophe correlations $\rho$).
- **5 Sensitivity Experiments:** (1) Demographic PRCC, (2) Genetic $B$ & $H_0$, (3) Catastrophe probability & spatial correlation $\rho$, (4) Carrying capacity $K$, (5) Combined intervention robustness matrix.
- **5-Layer Validation:** Unit tests (`pytest`), biological plausibility checks, historical trend consistency check (1967–2020 growth window), Vortex 10 specification & structural comparison bridge, and Monte Carlo uncertainty bounds (5th–95th percentiles).
