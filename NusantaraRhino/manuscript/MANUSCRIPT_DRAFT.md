# Evaluating the Long-Term Viability of the Javan Rhinoceros (*Rhinoceros sondaicus*) under Alternative Conservation Strategies: An Integrated Stochastic Population and Genetic Modeling Approach

**Manuscript Draft (Candidate for Scientific Publication)**  
*Prepared via the reproducible `NusantaraRhino` modeling framework*

---

## 1. Abstract
The Javan rhinoceros (*Rhinoceros sondaicus*) persists exclusively as a single small wild population in Ujung Kulon National Park (UKNP), Indonesia, rendering the species vulnerable to demographic stochasticity, genetic drift, and spatially concentrated catastrophic hazards. Building upon the 2015 Population and Habitat Viability Assessment (PHVA) and molecular evidence of historical mitochondrial lineage loss, we developed **NusantaraRhino**, an integrated individual-based stochastic model coupling life-history demography (enforcing realistic 15–16 month gestation and $\ge 3.5$-year inter-birth intervals via $\lceil 3.5 \rceil = 4$ discrete annual steps), 20-locus diploid Mendelian genetics with identity-by-descent inbreeding depression, and spatially correlated environmental catastrophes across a 100-year horizon (250 Monte Carlo replicates per scenario across a 5-scenario factorial design). Under baseline UKNP conditions (Scenario A: $N_0=76, K=80$, chronic drag, 2% annual catastrophe probability), mean population size declined to $19.30$ individuals (IQR $7.0–28.8$) with $54.0\%$ quasi-extinction ($N \le 15$; 95% Wilson CI: $47.8–60.1\%$) and $7.2\%$ total extinction ($4.6–11.1\%$), retaining $78.5\%$ of initial expected heterozygosity ($H_e = 0.4318$; $F_{\text{IBD}} = 0.1357$). Intensive single-site habitat and threat management (Scenario B: $K=105$, chronic drag removed, $+1.5\%$ survival protection) reduced total extinction to $0.0\%$ ($0.0–1.5\%$) and quasi-extinction to $0.8\%$ (mean $N_{100}=81.20$; $94.5\%$ $H_e$ retained). Under moderate UKNP management, establishing a second metapopulation (14 founders, periodic 10-year genetic exchange) eliminated total extinction under independent catastrophe risk (Scenario C1, $\rho=0.0$: $0.0\%$ extinction, mean $N_{100}=62.76$), whereas regional catastrophe correlation (Scenario C2, $\rho=0.6$) increased simultaneous dual-site shocks and raised extinction to $1.6\%$ ($F_{\text{IBD}}=0.0671$). Crucially, a factorial integrated strategy pairing full intensive *in situ* UKNP management with a hazard-independent second site (Scenario D) achieved the highest global abundance (mean $N_{100}=107.10$, median $109.5$, P5–P95: $56.5–155.0$), $0.0\%$ total extinction, $95.5\%$ $H_e$ retention ($0.5254$), and remained resilient even under pessimistic genetic load ($B=3.14, \text{catastrophe}=3\%/\text{yr}$: $0.0\%$ extinction, $1.25\%$ quasi-extinction). Latin Hypercube Sampling ($N=36$) with true Partial Rank Correlation Coefficients (PRCC) confirmed adult survival as the primary demographic driver ($\text{PRCC}=-0.8465$ with quasi-extinction, $+0.9278$ with $N_{100}, p<0.0001$), followed by calf survival ($\text{PRCC}=-0.6740$), conception probability ($\text{PRCC}=-0.6326$), and catastrophe probability ($\text{PRCC}=+0.4070, p=0.0208$).

---

## 2. Introduction
The Javan rhinoceros (*Rhinoceros sondaicus*) is a critically endangered megaherbivore confined to a single wild stronghold in Ujung Kulon National Park (UKNP), western Java. Single-site concentration near the Sunda Strait volcanic arc (Krakatau) exposes the entire species simultaneously to tsunami/volcanic hazards, disease outbreaks, invasive palm (*Arenga obtusifolia*) habitat encroachment, and small-population genetic erosion.

The 2015 CPSG Javan Rhino PHVA used Vortex to explore threats and second-population scenarios from a baseline of 58–61 individuals. Subsequently, museum-vs-modern ancient DNA analysis (Margaryan et al., 2020) documented the extinction of historical mitochondrial lineages, while IUCN SSC Asian Rhino Specialist Group synthesis reaffirmed slow reproductive biology (~15–16 month gestation, single calf, multi-year inter-birth intervals). More recently, BRIN (2026) and IUCN Translocation Guidelines emphasizes that any candidate second site requires rigorous ecological, social, management, and genetic connectivity assessment rather than assuming automatic demographic benefit.

To move beyond repeating earlier single-model exercises, this study introduces three specific contributions: (1) an open, reproducible Python individual-based model coupling explicit multi-year reproductive cooldowns, 20-locus Mendelian inheritance, and distinct chronic vs catastrophic pressures; (2) Gaussian-copula modeling of spatially correlated catastrophes ($P(C_1,C_2) \neq P(C_1)P(C_2)$); and (3) a factorial scenario design evaluated with Latin Hypercube Sampling (LHS), true Partial Rank Correlation Coefficients (PRCC), and multi-regime robustness matrices.

---

## 3. Materials and Methods
### 3.1 Parameter Provenance and Epistemic Guardrails
All 30 parameters were cataloged in `data/parameters.csv` (Table 1) with explicit `assumption_type` tags (`Literature-derived`, `Estimated`, `Assumed`, `Scenario-defined`) validated via Pydantic schemas (`src/models/parameters.py`). Unverified summary inbreeding estimates (e.g., unstandardized percentages without verified statistical definitions) were excluded as point inputs and instead bracketed via sensitivity experiments across initial heterozygosity ($H_0 \in [0.35, 0.75]$) and haploid lethal equivalents ($B \in [0.0, 3.14]$).

### 3.2 Factorial Experimental Design
Five core scenarios (Table 2; 100 years, 250 replicates, random seed `2026`, quasi-extinction floor $N \le 15$) were evaluated to separate *in situ* management intensity from spatial metapopulation structure:
- **A_Baseline_UKNP:** Single population ($N_0=76, K=80$, chronic drag $0.008/\text{yr}$, catastrophe $p=0.02$, severity $0.35$).
- **B_Managed_UKNP:** Single population with intensive habitat/protection ($K=105$, chronic drag $=0$, survival boost $+0.015$, catastrophe $p=0.018$, severity $0.30$).
- **C_Metapop_Uncorrelated ($\rho=0.0$):** Moderate UKNP management ($K_1=90$, drag $0.003$, boost $+0.008$) + second site ($K_2=70$) founded at year 5 with 14 individuals ($5\%$ handling mortality) and 2 managed migrants every 10 years.
- **C_Metapop_Correlated ($\rho=0.6$):** Identical to C1 except regional catastrophe correlation $\rho=0.6$.
- **D_Integrated_Managed_Metapop ($\rho=0.0$):** Full factorial combination pairing Scenario B's intensive *in situ* UKNP management ($K_1=105$, drag $=0$, boost $+0.015$) with Scenario C1's hazard-independent second site ($K_2=70$, 14 founders, 2 migrants/10 yr).

Five additional experiments quantified sensitivity to (1) demographic parameters via Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`, $n=36$) and exact OLS-residualized **Partial Rank Correlation Coefficients (PRCC)** alongside bivariate Spearman correlations, (2) genetic load $B \times H_0$, (3) catastrophe correlation $\rho \in \{0.0, 0.3, 0.6, 0.9\} \times \text{severity}$, (4) UKNP carrying capacity $K \in \{60, 80, 100, 120\}$, and (5) combined uncertainty regimes (Optimistic, Moderate, Pessimistic) across all 5 scenarios.

---

## 4. Model Development
The Python engine (`src/engines/`) integrates three coupled modules (Figure 1):
1. **Demographic IBM (`demographic.py`):** Tracks individual tuple $I_i=(a_i,s_i,b_i,g_i,\text{loc}_i,c_i)$. Females enter a 4-step discrete annual reproductive cycle ($\lceil 3.5 \rceil = 4$ years, enforcing 3 full post-calving blocked years) to respect 15–16 month gestation plus ~2-year lactation dependency. Density dependence acts near carrying capacity $K$.
2. **Population Genetics (`genetics.py`):** Tracks 20 unlinked diploid loci (`state_geno` for $H_o, H_e, A_R$; unique founder tags `ibd_geno` for true pedigree identity-by-descent $F_{\text{IBD}}$) and applies juvenile inbreeding depression $S_{\text{calf},0}\exp(-B \cdot F_i)$ across the dependent calf stage. Simulated $H_e$ trajectories are compared against Wright–Fisher expectations $H_t = H_0(1 - 1/(2N_e))^t$.
3. **Environmental Stochasticity (`environment.py`):** Separates annual Gaussian survival noise and chronic drag from bivariate Bernoulli catastrophes generated via Gaussian copula.

---

## 5. Results
### 5.1 Core Factorial Scenarios (Demography and Genetics)
Across 100 years (Table 3, Table 4, Figures 2–4):
- **Scenario A (Baseline UKNP)** showed substantial vulnerability (supporting **H1**): mean $N_{100}=19.30$ (P5–P95: $0.0–53.5$), $54.0\%$ quasi-extinction ($95\%$ Wilson CI: $47.8–60.1\%$), $7.2\%$ total extinction ($4.6–11.1\%$), $H_e$ declining to $0.4318$ ($78.5\%$ retained), and $F_{\text{IBD}}=0.1357$ (supporting **H3**).
- **Scenario B (Intensive UKNP Management)** strongly stabilized demography (supporting **H2**): mean $N_{100}=81.20$ (IQR $67.2–98.0$), $0.0\%$ total extinction ($0.0–1.5\%$), $0.8\%$ quasi-extinction, $94.5\%$ $H_e$ retained ($0.5195$), and low inbreeding ($F_{\text{IBD}}=0.0336$).
- **Scenario C1 ($\rho=0.0$)** achieved $0.0\%$ total extinction (mean $N_{100}=62.76$, quasi-extinction $6.0\%$, $H_e=0.5111$), whereas **Scenario C2 ($\rho=0.6$)** experienced $1.6\%$ total extinction and higher inbreeding ($F_{\text{IBD}}=0.0671$), confirming that regional hazard correlation erodes diversification gains (**H4 conditional**).
- **Scenario D (Integrated Intensive UKNP + Metapopulation $\rho=0.0$)** achieved the strongest global persistence: mean $N_{100}=107.10$ (median $109.5$, IQR $88.0–127.8$), $0.0\%$ total extinction, $1.2\%$ quasi-extinction, and the highest genetic diversity retention ($H_e=0.5254, 95.5\%$ retained; allelic richness $A_R=3.916$).

### 5.2 Sensitivity (LHS + True PRCC), Historical Check, and Cross-Model Bridge
- **Demographic LHS + True PRCC (Table 5, Figure 5B):** After controlling for covariates via PRCC, four parameters showed statistically significant partial rank correlations with quasi-extinction and final abundance: `adult_survival` ($\text{PRCC}_{\text{QE}}=-0.8465, \text{PRCC}_{N}=+0.9278, p<0.0001$), `calf_survival` ($\text{PRCC}_{\text{QE}}=-0.6740, \text{PRCC}_{N}=+0.8392, p<0.0001$), `annual_conception_prob` ($\text{PRCC}_{\text{QE}}=-0.6326, p=0.0001$), and `catastrophe_prob` ($\text{PRCC}_{\text{QE}}=+0.4070, p=0.0208; \text{PRCC}_{N}=-0.6793, p<0.0001$).
- **Genetic & Habitat Sensitivity (Exp 2 & Exp 4, Figure 6B):** Increasing $B$ from $0.0$ to $3.14$ (at $H_0=0.55$) raised baseline quasi-extinction from $28.75\%$ to $72.50\%$; raising UKNP $K$ from $60$ to $120$ halved quasi-extinction ($78.0\% \rightarrow 39.0\%$) and reduced $F_{\text{IBD}}$ from $0.2957$ to $0.0785$.
- **Spatial Catastrophe Correlation (Exp 3, Figure 5A):** Raising $\rho$ from $0.0$ to $0.9$ (at severity $0.35$) increased mean simultaneous dual-site catastrophes from $0.09$ to $1.40$ events per century.
- **Robustness Matrix (Exp 5, Figure 6A):** Under pessimistic load ($B=3.14$, catastrophe $3\%/\text{yr}$), unbuffered translocation under moderate UKNP protection (Scenarios C1–C2: $32.5–33.75\%$ quasi-extinction) underperformed intensive single-site management (Scenario B: $6.25\%$ quasi-extinction), whereas **Scenario D (Intensive UKNP + Metapopulation)** reduced total extinction to $0.0\%$ and quasi-extinction to $1.25\%$ (mean $N_{100}=72.53$).
- **Validation & Vortex 10 Bridge (Table 6):** A 50-year historical test ($N_0=30 \rightarrow 50\text{ yr}$) yielded mean $N_{50}=49.0$ (IQR $36.2–65.0$), consistent with documented Ujung Kulon growth. Parameter-matched Vortex 10 specifications (`results/vortex/vortex_scenario_spec.csv` & `.xml`) and a 2015 PHVA anchor ($N_0=60 \rightarrow$ mean $N_{100}=20.51$, $11.33\%$ extinction) are exported for external Vortex 10 execution without fabricating Vortex outputs.

---

## 6. Discussion
Our results align with classic long-lived mammal life-history theory: adult survival dominates population persistence, while chronic habitat compression ($K=60$–$80$) traps small populations in an accelerating drift–inbreeding feedback loop. Crucially, our factorial comparison of Scenarios B, C1/C2, and D resolves a major policy debate: translocation is not a substitute for *in situ* UKNP protection, because removing 14 founders without eliminating chronic pressure and mortality at the source weakens resilience under severe genetic load. However, once intensive *in situ* UKNP management is secured (Scenario B), adding a hazard-independent second population (Scenario D) breaks the single-site carrying capacity ceiling and provides the lowest tail risk under pessimistic uncertainty.

## 7. Conservation Implications
1. **Protect Adult Survival Above All:** Anti-poaching patrol, snare prevention, and health surveillance deliver the highest marginal persistence gain per unit effort ($\text{PRCC}=-0.8465$).
2. **Expand Functional Carrying Capacity in UKNP:** Invasive *Arenga* palm control and browse restoration directly suppress inbreeding accumulation ($F_{\text{IBD}}$ drops nearly fourfold from $K=60$ to $K=120$).
3. **Factorial Phased Second-Site Design (Scenario D):** Any second site must be built *on top of* intensive UKNP management, minimize hazard correlation ($\rho \rightarrow 0$, geographically independent from Sunda Strait volcanic/tsunami exposure), and maintain periodic genetic exchange.

## 8. Limitations
1. **Data Sparsity:** Nuclear genomic heterozygosity ($H_0$) and lethal equivalents ($B$) for *R. sondaicus* are not directly measured genome-wide; conclusions are therefore framed across explicit sensitivity ranges.
2. **No Exact Extinction-Date Prediction:** Stochastic PVA models estimate *conditional relative risk* under stated assumptions, not exact future extinction years.
3. **External Vortex Run Pending:** Vortex 10 binary is not installed in the local execution environment; cross-model comparison currently exports matched Vortex 10 specifications and awaits external Vortex CSV import.

## 9. Conclusion
Long-term survival of the Javan rhinoceros requires treating intensive *in situ* UKNP management and genetically managed, hazard-independent metapopulation expansion as **synergistic components of an integrated strategy (Scenario D)** rather than competing alternatives.

## 10. References
1. Miller, P. S., Lees, C., Ramono, W., Purwoto, A., Rubianto, A., Sectionov, Talukdar, B., & Ellis, S. (Eds.). (2015). *Population and Habitat Viability Assessment for the Javan Rhino: Briefing Book*. Apple Valley, MN: IUCN SSC Conservation Breeding Specialist Group (CPSG).
2. Margaryan, A., Sinding, M.-H. S., Liu, S., Vieira, F. G., Chan, Y. L., Hufthammer, A. K., ... & Gilbert, M. T. P. (2020). Recent mitochondrial lineage extinction in the critically endangered Javan rhinoceros (*Rhinoceros sondaicus*). *Zoological Journal of the Linnean Society*, 190(1), 372–383. https://doi.org/10.1093/zoolinnean/zlaa004
3. O'Grady, J. J., Brook, B. W., Reed, D. H., Ballou, J. D., Tonkyn, D. W., & Frankham, R. (2006). Realistic levels of inbreeding depression strongly affect extinction risk in wild populations. *Biological Conservation*, 133(1), 42–51. https://doi.org/10.1016/j.biocon.2006.05.016
4. Frankham, R., Bradshaw, C. J., & Brook, B. W. (2014). Genetics in conservation management: Revised recommendations for the 50/500 rules, Red List criteria and population viability analyses. *Biological Conservation*, 170, 56–63. https://doi.org/10.1016/j.biocon.2013.12.036
5. Marino, S., Hogue, I. B., Ray, C. J., & Kirschner, D. E. (2008). A methodology for performing global uncertainty and sensitivity analysis in systems biology (LHS-PRCC). *Journal of Theoretical Biology*, 254(1), 178–196. https://doi.org/10.1016/j.jtbi.2008.04.011
6. IUCN/SSC. (2013). *Guidelines for Reintroductions and Other Conservation Translocations* (Version 1.0). Gland, Switzerland: IUCN Species Survival Commission.
7. Lacy, R. C., & Pollak, J. P. (2023). *Vortex: A Stochastic Simulation of the Extinction Process* (Version 10.6.0). Brookfield, IL: Chicago Zoological Society / Species Conservation Toolkit Initiative (SCTI).
