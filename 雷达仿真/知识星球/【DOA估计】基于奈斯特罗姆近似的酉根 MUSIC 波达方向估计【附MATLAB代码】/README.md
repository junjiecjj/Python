# Nyström-Approximated Root-MUSIC for DOA Estimation

This repository contains an independent implementation and performance analysis of the **Nyström-approximated Root-MUSIC algorithm** for Direction-of-Arrival (DOA) estimation, based on the paper:

> Veerendra D. et al.,  
> *“Unitary Root-MUSIC Method With Nyström Approximation for 3-D Sparse Array DOA Estimation in Sensor Networks”*,  
> IEEE Sensors Letters, October 2024.

The work validates the key claims of the original paper regarding **computational efficiency** and **estimation accuracy**, and includes additional exploratory analysis.

---

## 📌 Project Overview

Subspace-based DOA estimation algorithms such as MUSIC and Root-MUSIC provide high angular resolution but suffer from high computational complexity due to eigenvalue decomposition of large covariance matrices.  
The Nyström approximation reduces this cost by approximating the signal subspace using a **subset of sensors**, significantly lowering runtime while preserving estimation performance.

This repository provides:
- Baseline MUSIC implementations
- Nyström-approximated MUSIC implementations
- Cramér–Rao Bound (CRB) benchmarking
- Computational time analysis
- An exploratory clustering-based post-processing extension

---

## 📂 Repository Structure

### Core Implementations
- `MUSIC_No_Nystrom_1D.m`  
  Baseline 1-D MUSIC implementation (reference method).

- `MUSIC_Nystrom_Approximation_1D.m`  
  1-D MUSIC with Nyström approximation.

- `MUSIC_2D_No_Nystrom.m`  
  Baseline 2-D MUSIC for azimuth–elevation DOA estimation.

- `MUSIC_Nystrom_No_KNN.m`  
  2-D Nyström-approximated MUSIC (core reproduction of the original method).

- `nystromMusicDOA.m`  
  Generalized Nyström-based DOA estimation function used for experiments.

---

### Analysis and Utilities
- `CRB_for_MUSIC_NystromMusic.m`  
  Computes RMSE vs SNR and compares MUSIC and Nyström-MUSIC performance against the Cramér–Rao Bound.

- `findTopTwoLocalMaxima.m`  
  Utility function for peak detection in MUSIC spectra.

- `time_complexity.csv`  
  Runtime measurements used for computational complexity comparison.

---

### Exploratory Extension (Not in Original Paper)
- `MUSIC_2D_Nystrom_KNN.m`  
  Adds a **clustering-based post-processing step** (k-means) to aggregate spectral peaks for more robust DOA estimation in noisy scenarios.  
  ⚠️ This is an **exploratory enhancement** and **not part of the original paper**.

---

## ▶️ How to Run

1. Open MATLAB and set the repository root as the current folder.
2. Run individual scripts depending on the experiment:

### Baseline MUSIC
```matlab
MUSIC_No_Nystrom_1D
MUSIC_2D_No_Nystrom
