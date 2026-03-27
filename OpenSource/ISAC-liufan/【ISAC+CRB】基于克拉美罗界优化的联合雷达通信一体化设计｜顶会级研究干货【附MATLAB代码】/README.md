# CRB-ISAC Beamforming Reproduction

**Baseline P1-C** in [Awesome-ISAC](https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications)

> 📖 Part of [Yuanhao Cui](https://yuanhao-cui.info/)'s ISAC paper-repro infrastructure

---

## 📄 Paper

**Title**: Cramer-Rao Bound Optimization for Joint Radar-Communication Design

**Authors**: Fan Liu, Ya-Feng Liu, Ang Li, Christos Masouros, Yonina C. Eldar

**Venue**: IEEE Transactions on Signal Processing (TSP), Vol. 70, pp. 240–253, 2022

**Awards**: 🏆 2024 IEEE SPS Best Paper Award

**DOI**: [10.1109/TSP.2021.3135692](https://doi.org/10.1109/TSP.2021.3135692)

---

## 🔬 What This Baseline Reproduces

| Figure | Description | Status |
|--------|-------------|--------|
| **Fig. 2** | Root-CRB / MSE vs SINR threshold (K=1) | ✅ |
| **Fig. 3** | Beampattern comparison (K=4, SINR=15dB) | ✅ |

### Key Results

- **Fig. 2**: Validates the closed-form solution for single-user CRB minimization. Shows that estimation errors remain stable when SINR requirements are below 30 dB, then rise sharply.
- **Fig. 3**: Demonstrates that the proposed CRB-Min design achieves the highest power at the target angle (0°) compared to benchmark beampattern approximation methods.

---

## 🧮 Mathematical Background

### System Model

MIMO DFRC BS with $N_t$ transmit, $N_p$ receive antennas, serving $K$ users while detecting a point target.

**CRB for point target angle** $\theta$:

$$\text{CRB}(\theta) = \frac{1}{2L \cdot \text{SNR}_{\text{eff}}}$$

where $L$ is the frame length and $\text{SNR}_{\text{eff}}$ is the effective radar SNR, which depends on the power allocated to radar after satisfying communication SINR constraints.

**Power sharing model**: Given total power $P_r$ and per-user SINR threshold $\gamma_k$:

$$P_{\text{radar}} = P_r - \sum_k \gamma_k \sigma^2$$

### Problem Formulation

**Problem (P1) — CRB minimization with SINR constraints**:

$$\min_{\mathbf{W}} \text{CRB}(\theta) \quad \text{s.t.} \quad \text{SINR}_k \geq \gamma_k, \forall k, \quad \|\mathbf{W}\|_F^2 \leq P_r$$

Solved via **Semidefinite Relaxation (SDR)** + rank-one extraction (Theorem 4 in paper).

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy cvxpy matplotlib scipy pytest pytest-cov

# Reproduce Fig. 2 (Root-CRB vs SINR)
python examples/reproduce_fig2.py

# Reproduce Fig. 3 (Beampattern comparison)
python examples/reproduce_fig3.py

# Run tests
pytest tests/ -v
```

---

## 📁 Project Structure

```
.
├── src/
│   ├── model.py      # CRB model, system params, steering vectors
│   ├── solver.py     # SDR solver, closed-form, benchmark designs
│   └── metrics.py    # Beampattern, RMSE, MSE, trend verification
├── tests/
│   ├── test_model.py
│   ├── test_solver.py
│   ├── test_metrics.py
│   └── test_reproducibility.py
├── examples/
│   ├── reproduce_fig2.py
│   └── reproduce_fig3.py
├── configs/
│   └── default.yaml
├── README.md
└── requirements.txt
```

---

## 📊 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N_t$ | 16 | Transmit antennas |
| $N_p$ | 20 | Receive antennas |
| $K$ | 4 | Communication users |
| $L$ | 30 | Frame length |
| $P_r$ | 30 dBm | Total transmit power |
| $\sigma^2$ | 0 dBm | Noise variance |
| $\gamma$ | 15 dB | Per-user SINR threshold |

---

## ✅ Test Results

```bash
pytest tests/ -v
# test_model.py       ✅ 6 tests
# test_solver.py      ✅ 9 tests
# test_metrics.py     ✅ 6 tests
# test_reproducibility.py ✅ 6 tests
```

---

## 🤝 Contributing

This baseline follows the [Awesome-ISAC contributing guidelines](https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications/blob/main/CONTRIBUTING.md).

To add this baseline to Awesome-ISAC, see the Reproducible Baselines section in the main README.

---

## 📝 Citation

```bibtex
@article{liu2022crb,
  author={Liu, Fan and Liu, Ya-Feng and Li, Ang and Masouros, Christos and Eldar, Yonina C.},
  title={Cramer-Rao Bound Optimization for Joint Radar-Communication Design},
  journal={IEEE Trans. Signal Processing},
  volume={70},
  pages={240--253},
  year={2022},
  doi={10.1109/TSP.2021.3135692}
}
```
