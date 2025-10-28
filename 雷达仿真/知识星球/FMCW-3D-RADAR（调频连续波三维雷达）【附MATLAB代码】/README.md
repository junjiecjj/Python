# 🌌 FMCW-3D-RADAR

**A MATLAB Simulation and Visualization Platform for 3D FMCW Radar (Range–Velocity–Angle Estimation)**

## 📘 Overview

**FMCW-3D-RADAR** is a fully self-contained MATLAB project that demonstrates **range, velocity, and angle estimation** in Frequency-Modulated Continuous-Wave (FMCW) radar systems.

The project builds the entire processing chain — from waveform generation and dechirp modeling to FFT-based range–Doppler processing, 2D-CFAR detection, and array-based angle estimation — with intuitive 2D/3D visualizations.

It aims to help researchers and engineers understand, simulate, and validate **FMCW radar sensing principles** and **signal processing algorithms** in a unified framework.

---

## 🧩 Key Features

* **Signal Modeling**: Supports single-TX, multi-RX uniform linear array (ULA) configurations.
* **Target Scene Definition**: Multiple targets with configurable range, velocity, and azimuth.
* **Range–Doppler Processing**: 2D FFT-based detection and visualization.
* **Adaptive Detection**: 2D CA-CFAR for thresholding and peak extraction.
* **Angle Estimation**: FFT beamforming and high-resolution MUSIC algorithm.
* **3D Visualization**:

  * Range–Doppler Map (with detection & truth overlay)
  * Angle–Range Map (strongest-Doppler slice per range)
  * 3D Range–Velocity–Angle scatter plot
* **Engineering Metrics**: Automatically computes unambiguous range/velocity and prints detected targets with power ranking.

---

## ⚙️ Processing Flow

```text
TX Chirp Generation
        ↓
Target Delay + Doppler Modeling
        ↓
Dechirp (Beat Signal) + ADC Sampling
        ↓
Fast-time FFT → Range Bins
        ↓
Slow-time FFT → Doppler Bins
        ↓
2D CA-CFAR Detection
        ↓
Angle Estimation (FFT / MUSIC)
        ↓
RA Map & 3D Visualization
```

---


## 🧠 Applications

* Academic study of FMCW radar signal processing
* Rapid prototyping of MIMO / automotive radar algorithms
* ISAC (Integrated Sensing and Communication) research
* Pre-FPGA or SDR algorithm validation

---

## 🖥️ Environment

* MATLAB R2020b or newer (no extra toolboxes required)
* 64-bit OS, 8 GB+ RAM recommended
* Optional GPU acceleration supported (`gpuArray`)

---




