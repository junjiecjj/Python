# 3D MIMO Beamforming using MUSIC Algorithm

## 📡 Project Overview

This project implements a complete 3D Direction of Arrival (DOA)
estimation pipeline using a planar MIMO antenna array.

The system compares classical Bartlett beamforming with the
high-resolution MUSIC (Multiple Signal Classification) algorithm to
demonstrate the resolution advantage of subspace-based methods over
conventional beamforming.

The implementation is modular and structured to reflect a practical
radar signal processing architecture.

------------------------------------------------------------------------

## 🔬 Features

-   Planar 4×4 MIMO array simulation
-   Narrowband signal modeling for multiple targets
-   Sample covariance matrix estimation
-   Eigenvalue decomposition for signal/noise subspace separation
-   2D MUSIC spatial spectrum computation
-   Classical Bartlett beamformer implementation
-   Automatic peak detection for DOA estimation
-   3D spatial spectrum comparison visualization
-   Additional analytical plots:
    -   Top-view heatmap
    -   Azimuth slice
    -   Eigenvalue spectrum

------------------------------------------------------------------------

## 🎯 Key Insight

Classical beamforming produces wider lobes and limited angular
resolution.

MUSIC, based on signal/noise subspace separation, achieves significantly
sharper peaks and superior angular resolution --- particularly when
targets are closely spaced.

The side-by-side comparison clearly illustrates this performance
difference.

------------------------------------------------------------------------

## 🧠 Algorithms Implemented

-   Sample Covariance Matrix Estimation
-   Eigenvalue Decomposition
-   Signal and Noise Subspace Separation
-   Noise Subspace Projection (MUSIC)
-   2D Grid-Based Spatial Spectrum Scanning
-   Bartlett (Conventional) Beamforming
-   Peak Detection for DOA Estimation

------------------------------------------------------------------------

## ⚙️ System Configuration

-   Carrier Frequency: 77 GHz
-   Array Type: 4×4 Planar Rectangular Array
-   Element Spacing: λ/2
-   Number of Targets: 2
-   SNR: 20 dB
-   Scan Grid:
    -   Azimuth: -90° to 90°
    -   Elevation: -30° to 30°

------------------------------------------------------------------------

## 📁 Project Structure
mimo_3d_beamforming/\
├── config.py               \
├── array_geometry.py       \
├── signal_simulator.py     \
├── covariance.py           \
├── music_2d.py             \
├── beamformer.py           \
├── peak_detection.py       \
├── visualization.py        \
├── main.py                 \
└── README.md

------------------------------------------------------------------------

## ▶️ How to Run

Install required libraries:

          pip install numpy matplotlib

Run the simulation:

          python main.py

------------------------------------------------------------------------
## 📊 Output (Comparison Visualization)

Here’s a high-resolution sample output from the system:

<img width="1444" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/715865cc-31c4-4ac6-a58d-755d67330223" />

------------------------------------------------------------------------

## 📌 Applications

-   Automotive Radar Signal Processing
-   3D Direction of Arrival Estimation
-   Array Signal Processing Research
-   Subspace-Based Super-Resolution Methods

------------------------------------------------------------------------

## 👨‍💻 Author

Developed as part of advanced exploration in array signal processing and
high-resolution DOA estimation.
