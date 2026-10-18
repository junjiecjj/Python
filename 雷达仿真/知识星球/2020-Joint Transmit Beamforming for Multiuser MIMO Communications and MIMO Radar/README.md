# ISAC Anti-Jamming Transmit Beamforming

MATLAB implementation of anti-jamming joint transmit beamforming for MIMO Integrated Sensing and Communication (ISAC) systems.

## Overview

This repository contains the simulation code for the paper:

> **"Anti-Jamming Transmit Beamforming for Air-Ground Cooperative MIMO ISAC Systems"**

A ground base station (BS) with a uniform linear array simultaneously performs radar sensing toward aerial targets and downlink communication to ground users. A malicious aerial jammer is suppressed via explicit null-steering in the transmit beampattern.

## Core Algorithms

| File | Description |
|------|------------|
| `waveform_design_radar_only_covmat.m` | Radar-only covariance design (upper bound) |
| `waveform_design_SDR_covmat.m` | SDR joint beamforming (baseline, Liu et al. 2018) |
| `waveform_design_ZF_covmat.m` | ZF joint beamforming (baseline) |
| `waveform_design_QTFP.m` | QT-FP iterative beamforming (baseline, Shen et al. 2024) |
| `waveform_design_SDR_anti_jamming.m` | **SDR-AJ: Proposed anti-jamming SDR with null constraint** |
| `waveform_design_SDR_anti_jamming_multi.m` | **SDR-AJ Multi-Jammer: J simultaneous nulls** |
| `waveform_design_QTFP_anti_jamming.m` | QT-FP Anti-Jamming with soft null penalty |
| `waveform_design_WMMSE.m` | WMMSE iterative beamforming |
| `waveform_design_WMMSE_anti_jamming.m` | WMMSE Anti-Jamming with null regularization |

## Utility Functions

| File | Description |
|------|------------|
| `ULA_steering_vector.m` | ULA steering vector |
| `rician_channel.m` | Air-ground Rician channel model |
| `eval_null_depth.m` | Null depth evaluation |
| `SUM_RATE_func.m` | Sum-rate computation |
| `INR_func.m` | Interference-to-noise ratio |
| `nearestSPD.m` | Nearest SPD projection (Higham 1988) |
| `QPSK_mapper.m` / `QPSK_demodulation.m` | QPSK modulation/demodulation |

## Simulation Scripts

| File | Description |
|------|------------|
| `fast_sim.m` | **Main simulation**: Fig 1 (beampattern), Fig 2 (SINR vs jammer power + ISAC efficiency), Fig 3 (robustness stress tests) |
| `main.m` | Original paper: MSE vs SINR, sum-rate comparisons |
| `beamPatternFigure.m` | Beampattern comparison figure |
| `matlab_setup.m` | MATLAB path + CVX setup |

## Requirements

- MATLAB R2022b or later
- [CVX](http://cvxr.com/cvx/) with SeDuMi SDP solver

## Quick Start

```matlab
% 1. Setup CVX
cd('path/to/cvx'); cvx_setup;

% 2. Add code to path
cd('path/to/repo'); addpath(genpath(cd));

% 3. Run main simulation
fast_sim
```

## Key Results

- **Null depth**: > 25 dB at jammer direction (17.6 dB improvement over SDR baseline)
- **Rate loss**: < 5% (anti-jamming penalty)
- **MSE increase**: < 0.002 (beampattern fidelity maintained)
- **SINR gain**: ~10 dB at 30 dBm jammer power

## References

- [1] Liu et al., "MU-MIMO communications with MIMO radar," IEEE TWC, 2018.
- [18] Shen et al., "Accelerating quadratic transform and WMMSE," IEEE JSAC, 2024.
