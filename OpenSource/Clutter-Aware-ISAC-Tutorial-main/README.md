# Clutter-Aware Integrated Sensing and Communication: Models, Methods, and Future Directions

This repository contains the MATLAB simulation code for the paper:

> R. Liu, P. Li, M. Li, and A. L. Swindlehurst, "Clutter-aware integrated sensing and communication: Models, methods, and future directions," *Proc. IEEE*, to appear, 2026.
> [[arXiv]](https://arxiv.org/abs/2602.10537)

## Overview

We develop a unified wideband MIMO-OFDM signal model that captures both **cold clutter** (environmental backscatter of the probing waveform) and **hot clutter** (external interference reflections) across the space, time, and frequency domains. The code implements the complete simulation framework including:

- **Clutter channel generation**: cold clutter with frequency-correlated coefficients, hot clutter via bistatic scattering paths, and UAV-like extended targets (Section IV)
- **Slow-time clutter suppression**: symbol-wise averaging (Avg.), recursive mean averaging (RMA), and consecutive symbol differencing (CSD) (Section V-A)
- **Spatial-domain suppression**: MVDR beamforming under ideal, insufficient-training, and AoA-mismatch scenarios (Section V-B)
- **Space-time adaptive processing (STAP)**: conventional, full-rank, and reduced-rank STAP for joint angle-Doppler filtering (Section V-C)
- **Clutter-aware transceiver co-design**: BLP-based alternating optimization with Dinkelbach-SDP for joint transmit covariance and receive beamformer design under communication QoS constraints (Section VI)
- **Clutter kernel estimation**: probing-based estimation of the spatial clutter kernel and its projection onto the completely positive (CP) cone (Section IV)

## Requirements

- **MATLAB** R2022a or later (tested on R2023a)
- **CVX** (version 2.2 or later) with a compatible SDP solver - [http://cvxr.com/cvx/](http://cvxr.com/cvx/)
- **MOSEK** (recommended, used in the Dinkelbach-SDP solver) - [https://www.mosek.com/](https://www.mosek.com/)
  - If MOSEK is unavailable, change `cvx_solver mosek` to `cvx_solver sdpt3` or `cvx_solver sedumi` in `function/solve_TX_Dinkelbach.m`
- **Parallel Computing Toolbox** (optional but recommended)
  - Several scripts use `parfor` for Monte Carlo acceleration. If the toolbox is unavailable, replace `parfor` with `for` - the code will run correctly but more slowly.

## Repository Structure

```
code/
├── README.md                       # This file
├── param_basic.m                   # System parameter initialization (run first)
├── Vcc_Reta_esti.m                 # Clutter kernel and covariance estimation
├── BLP_BFstap_Design.m             # BLP-based joint transceiver design
│
├── fig2_RDM_derandom.m             # Fig. 2:  Range-Doppler maps (de-randomization)
├── fig4_5_slow_time_Processing.m   # Fig. 4-5: Slow-time suppression + MUSIC spectra
├── fig8_spatial_MVDR.m             # Fig. 8:  Spatial MVDR beampatterns
├── fig10_STAP_sim.m                # Fig. 10: Angle-Doppler maps (STAP)
├── fig12a_SCNR_Ptx.m               # Fig. 12a: SCNR vs. transmit power
├── fig12b_SCNR_Gamma.m             # Fig. 12b: SCNR vs. communication SINR threshold
│
├── function/                       # Supporting functions
│   ├── gen_clutter_channels.m      # Generate cold/hot/UAV clutter channel tensors
│   ├── draw_H_cold_realization.m   # Draw one random cold clutter realization
│   ├── echo_tar.m                  # Generate target echo signal
│   ├── clutter_reflectivity_GIT.m  # GIT-type empirical clutter reflectivity model
│   ├── scatterAngles.m             # Generate scatterer angles with target avoidance
│   │
│   ├── symbol_wise_mean_esti.m     # Slow-time: symbol-wise mean subtraction (Avg.)
│   ├── rma_filter.m                # Slow-time: recursive mean averaging filter (RMA)
│   ├── frame_wise_difference.m     # Slow-time: consecutive symbol differencing (CSD)
│   ├── music_spectrum.m            # MUSIC spatial spectrum computation
│   ├── mvdr_beamformer.m           # MVDR receive beamformer with diagonal loading
│   │
│   ├── kernel_apply.m              # Apply clutter kernel: Rcc = Vcc * vec(Rxx)
│   ├── build_Ac.m                  # Build clutter-dependent matrix for TX optimization
│   ├── project_Vcc_to_CP.m         # Project clutter kernel onto the CP cone (via Choi matrix)
│   ├── compute_SCNR.m              # Compute overall SCNR across subcarriers
│   ├── comp_SINR_comm.m            # Compute per-user communication SINR
│   │
│   ├── solve_BLP_AO.m              # ISAC: AO-based joint TX/RX design (BLP + MVDR)
│   ├── solve_Radar_AO.m            # Radar-only: AO-based TX/RX design
│   ├── solve_TX_Dinkelbach.m       # Dinkelbach-SDP solver (ISAC with QoS constraints)
│   ├── solve_TX_Dinkelbach_radar.m # Dinkelbach-SDP solver (radar-only)
│   ├── baseline_comm_heur.m        # Comm-only and heuristic ISAC baselines
│   ├── recover_beams.m             # Recover beamformers from covariance solutions
│   │
│   ├── annotRectData.m             # Annotation: rectangle in data coordinates
│   ├── annotEllipseData.m          # Annotation: ellipse in data coordinates
│   └── localData2Norm.m            # Convert data coordinates to normalized figure units
│
└── data/                           # Generated data  
    ├── parameters_basic.mat        # Output of param_basic.m
    └── covMat_BLP.mat              # Output of Vcc_Reta_esti.m
```

## Quick Start

### Step 1: Generate basic parameters

```matlab
run('param_basic.m')
```

This creates `data/parameters_basic.mat` containing the MIMO-OFDM system configuration, target parameters, hot source settings, and clutter region geometry.

### Step 2: Generate individual figures

The following scripts can be run independently after Step 1:

| Script | Paper Figure | Description |
|--------|-------------|-------------|
| `fig2_RDM_derandom.m` | Fig. 2 | Range-Doppler maps with and without de-randomization |
| `fig4_5_slow_time_Processing.m` | Fig. 4 & 5 | MUSIC spectra and RDMs under slow-time suppression |
| `fig8_spatial_MVDR.m` | Fig. 8 | MVDR beampatterns (ideal, insufficient training, AoA mismatch) |
| `fig10_STAP_sim.m` | Fig. 10 | Angle-Doppler maps (conventional, full-rank STAP, reduced-rank STAP) |

### Step 3: Run BLP-based transceiver design (requires CVX)

```matlab
% Step 3a: Estimate clutter kernel and covariance matrices (compute-intensive)
run('Vcc_Reta_esti.m')

% Step 3b: Run the joint transceiver design
run('BLP_BFstap_Design.m')
```

### Step 4: Reproduce SCNR performance curves (requires CVX, compute-intensive)

```matlab
% Requires covMat_BLP.mat from Step 3a
run('fig12a_SCNR_Ptx.m')     % Fig. 12a: SCNR vs. transmit power
run('fig12b_SCNR_Gamma.m')   % Fig. 12b: SCNR vs. communication SINR threshold
```

> **Note**: `Vcc_Reta_esti.m`, `fig12a_SCNR_Ptx.m`, and `fig12b_SCNR_Gamma.m` involve large-scale Monte Carlo simulations and may take several hours to complete. Using `parfor` with multiple workers is highly recommended.

## System Parameters

The default parameters in `param_basic.m` correspond to a 28 GHz mmWave MIMO-OFDM ISAC system:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Carrier frequency | 28 GHz | mmWave band |
| Subcarrier spacing | 120 kHz | 5G NR numerology μ = 3 |
| Number of subcarriers | 512 | Total bandwidth ≈ 61.44 MHz |
| Number of OFDM symbols | 256 | Coherent processing interval |
| Transmit/Receive antennas | 16 / 16 | ULA with half-wavelength spacing |
| Transmit power | 53 dBm | |
| Communication users | 3 | |
| Target RCS | -13 dBsm | Point target |
| Clutter scatterers | 100 | Distributed in two range rings |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2026clutter,
  author  = {Liu, Rang and Li, Peishi and Li, Ming and Swindlehurst, A. Lee},
  title   = {Clutter-Aware Integrated Sensing and Communication: Models, Methods, and Future Directions},
  journal = {Proc. IEEE},
  year    = {2026},
  note    = {to appear}
}
```

## Contact

- **Rang Liu** - University of California, Irvine - rangl2@uci.edu
- **Peishi Li** - Dalian University of Technology - lipeishi@mail.dlut.edu.cn

## License

This code is provided for academic and research purposes. Please contact the authors for commercial use.
