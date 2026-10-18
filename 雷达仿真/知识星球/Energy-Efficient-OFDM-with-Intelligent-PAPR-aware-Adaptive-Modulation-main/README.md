# Energy Efficient OFDM with Intelligent PAPR-Aware Adaptive Modulation

## Overview

This repository contains the implementation of a novel **PAPR-aware adaptive modulation** scheme to enhance the energy efficiency of Orthogonal Frequency Division Multiplexing (OFDM) systems. The scheme mitigates the high **Peak-to-Average Power Ratio (PAPR)** of OFDM signals using an intelligent, deep learning-based approach, optimizing both modulation and power allocation per subcarrier.

---

## Key Features

- **PAPR-Aware Adaptive Modulation**:
  - Modulation and power allocation optimized based on PAPR awareness to improve the Power Amplifier (PA) efficiency and reduce energy consumption.
  
- **Deep Learning-Based Optimization**:
  - A deep neural network (DNN) is used to solve the non-convex optimization problem caused by PAPR.
  - Two learning approaches are implemented: **Online learning** and **Unsupervised learning**.

- **Energy Efficiency Gains**:
  - Achieves up to **3 dB** improvement in energy efficiency over conventional PAPR-unaware OFDM systems.
  - Lowers PAPR by up to **3 dB**, especially in high Signal-to-Noise Ratio (SNR) scenarios.

- **Supports Multiple Modulation Orders**:
  - The system supports modulation orders like **4-QAM**, **16-QAM**, **64-QAM**, and higher, allowing flexibility based on channel conditions.

---

## System Model

This system is designed for a point-to-point communication link between an **Access Point (AP)** and a user, utilizing **K** subcarriers. Each subcarrier can use a different modulation scheme based on channel conditions, while power allocation is dynamically optimized across subcarriers to reduce PAPR and improve energy efficiency.

### Key Parameters:
- **OFDM Transmission**: Multi-carrier modulation scheme for transmitting data over subcarriers.
- **PAPR**: Peak-to-Average Power Ratio, a major challenge in OFDM systems.
- **Power Allocation**: Dynamically adjusts power per subcarrier to minimize energy consumption and maximize PA efficiency.
- **Deep Learning**: A DNN is employed to learn the optimal power and modulation settings for maximizing energy efficiency.

---

## DNN schemes
The primary objective is to maximize the energy efficiency, which is the ratio of throughput to total consumed power. The optimization problem is tackled using one of the DNN schemes below, which solves for the optimal power allocation and modulation scheme under non-convex constraints.

### 1. Online Learning

An **online learning** algorithm is implemented to dynamically adapt the power allocation and modulation per subcarrier based on **instantaneous Channel State Information (CSI)**. A deep neural network is trained during the transmission process to continuously improve energy efficiency.
- Input: Channel state information (CSI) for each subcarrier.
- Output: Power allocation \( \mathbf{p} \) and modulation indices \( \mathbf{a} \).
- Activation functions: **Softmax** is used for power allocation, and **Gumbel-Softmax** for selecting modulation orders.

### 2. Unsupervised Learning

The **unsupervised learning** approach relies on random channel realizations and batch training to generalize the system to unseen conditions. It is less specific than online learning but provides robust solutions across a range of channel states.
- Input: Channel state information (CSI) for each subcarrier.
- Output: Power allocation \( \mathbf{p} \) and modulation indices \( \mathbf{a} \).
- Activation functions: **Softmax** is used for power allocation, and **Gumbel-Softmax** for selecting modulation orders.

---

## Configuration

You can configure system parameters such as:
- Number of subcarriers (K)
- Modulation orders (e.g., 64-QAM)
- Maximum power constraints
- Path loss, noise levels, and SNR values

---

## Results

### Energy Efficiency

- The proposed system shows up to **3 dB improvement** in energy efficiency over conventional PAPR-unaware OFDM systems.
- The performance enhancement is particularly significant at higher SNRs, where PAPR becomes a limiting factor.

### PAPR Reduction

- The PAPR-aware scheme lowers PAPR by up to **3 dB** compared to a PAPR-unaware system, especially in high-SNR environments, contributing to higher PA efficiency.

---

## Contributing

Contributions are welcomed! Feel free to fork the repository, create a new branch, and submit a pull request with improvements or new features.

---

## References

The algorithm and methodology are based on the following research paper:

- N. A. Mitsiou, P. D. Diamantoulakis, P. G. Sarigiannidis and G. K. Karagiannidis, "Energy Efficient OFDM With Intelligent PAPR-Aware Adaptive Modulation," in IEEE Communications Letters, vol. 27, no. 12, pp. 3290-3294, Dec. 2023, doi: 10.1109/LCOMM.2023.3324137.

## License

- The code in this repository is licensed under the MIT License. See the LICENSE file for details.
- The paper and any associated figures or text are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
