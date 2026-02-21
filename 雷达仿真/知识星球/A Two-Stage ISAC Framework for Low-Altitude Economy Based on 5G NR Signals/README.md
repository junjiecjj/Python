# Baseline Algorithms for "A Two-Stage ISAC Framework for Low-Altitude Economy Based on 5G NR Signals"

This repository provides implementations of baseline algorithms used for performance comparison in the paper:

> **"A Two-Stage ISAC Framework for Low-Altitude Economy Based on 5G NR Signals"**

These methods represent classical and state-of-the-art approaches for joint range–Doppler or angle estimation in Integrated Sensing and Communication (ISAC) systems based on 5G NR signals.

## Included Baseline Algorithms

1. 2D-FFT  
A conventional two-dimensional Fast Fourier Transform method for estimating range and Doppler (or angle). It is widely adopted due to its simplicity and low computational complexity.

> **Reference:**  
> L. Pucci, E. Paolini, and A. Giorgetti, “System-level analysis of joint sensing and communication based on 5G new radio,” *IEEE J. Sel. Areas Commun.*, vol. 40, no. 7, pp. 2043–2055, 2022.

2. 2D-MUSIC  
A high-resolution subspace-based algorithm that extends the MUSIC method to two dimensions. It achieves finer resolution than FFT-based approaches, especially in low-SNR regimes.

> **Reference:**  
> R. Xie, D. Hu, K. Luo, and T. Jiang, “Performance analysis of joint range-velocity estimator with 2D-MUSIC in OFDM radar,” *IEEE Trans. Signal Process.*, vol. 69, pp. 4787–4800, 2021.

3. 2D CS-AN (Compressive Sensing with Atomic Norm)  
A super-resolution method based on atomic norm minimization that enables gridless parameter estimation in continuous delay–Doppler domains. 

> **Reference:**  
> L. Zheng and X. Wang, “Super-resolution delay-doppler estimation for OFDM passive radar,” *IEEE Trans. Signal Process.*, vol. 65, no. 9, pp. 2197–2210, 2017.

## Purpose

These implementations serve as reference baselines to fairly evaluate the proposed two-stage ISAC framework under identical simulation conditions (e.g., 5G NR waveform, channel model, SNR settings).

