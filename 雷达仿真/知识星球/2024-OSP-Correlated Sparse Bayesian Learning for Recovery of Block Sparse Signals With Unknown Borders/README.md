# Correlated Sparse Bayesian Learning (CorSBL) — MATLAB Code

This repository contains the MATLAB code behind:

**Didem Doğan and Geert Leus**,  
*Correlated Sparse Bayesian Learning for Recovery of Block Sparse Signals With Unknown Borders*,  
**IEEE Open Journal of Signal Processing**, 5:421–435, 2024.  
DOI: https://doi.org/10.1109/OJSP.2024.3360914

The code reconstructs complex block-sparse signals with **unknown block borders** and compares the **proposed CorSBL** algorithm against established SBL families.

---

## What’s implemented

### Algorithms
- **CorSBL (proposed)**: `MSBL_correlated.m`  
  Correlation-aware SBL with a **tridiagonal** coefficient covariance (neighbor coupling).
- **SBL, PCSBL and CSBL** (pattern-coupled): `MPCSBL.m` (SBL and PCSBL), `MPCSBL_alternative.m`(CSBL), 
- **BSBL (block SBL)**: `BSBL_EM.m`, `BSBL_FM.m` (evaluated with **h = 2** and **h = 4**)
- **EBSBL (extended BSBL)**: `EBSBL_BO.m` (evaluated with **h = 2** and **h = 4**)

> Throughout the repository, CorSBL is compared to **BSBL(h=2)**, **BSBL(h=4)**, **EBSBL(h=2)**, **EBSBL(h=4)**, **SBL**, **PCSBL**, and **CSBL**.

---

## Main experiment scripts

- **Monte-Carlo plots / sweeps**: files starting with `nmse_performance_*.m`  
  Produce figures over varying conditions (e.g., \(m/n\), SNR, etc.).  
  Metrics include **NMSE** and, in some scripts, **success rate** and related indicators.
- **Single-trial diagnostics**: files starting with `EM_Methods_*.m`  
  Run one Monte-Carlo realization (no plot aggregation) for a specific scenario to inspect reconstructions and scalar NMSE values.

---

## Experiment model (common pattern)

1. **Signal**: complex, block-sparse vector \(x \in \mathbb{C}^n\) with **unknown block starts/lengths**; blocks may be **correlated** internally.
2. **Measurements**: \( y = A x + n \), where columns of \(A\) are typically **\(\ell_2\)-normalized**.  
   Noise level is set via **SNR (dB)** and added as complex Gaussian.
3. **Comparisons**:  
   CorSBL vs **PCSBL**, **CSBL**, **BSBL** (*h = 2, 4*), **EBSBL** (*h = 2, 4*).
4. **Metrics**: **NMSE** (and, in some scripts, **success rate** thresholds), averaged over MC trials.

---

## Notes

- BSBL/EBSBL runs are explicitly evaluated with **block lengths h = 2** and **h = 4**, matching the paper’s comparisons.
- PCSBL/CSBL implementations use pattern-coupling  modeling without fixing block borders.
- CorSBL employs **neighbor-coupled hyperparameters** via a **tridiagonal** prior; updates are derived to keep computation efficient.

---

## Citation

If you use this code, please cite:

> D. Doğan and G. Leus, “Correlated Sparse Bayesian Learning for Recovery of Block Sparse Signals With Unknown Borders,” **IEEE OJSP**, 2024, doi:10.1109/OJSP.2024.3360914.

---

## Contact

**Didem Doğan Başkaya**  
LinkedIn: https://www.linkedin.com/in/didemdoganbaskaya  
GitHub: https://github.com/Didemld
