# Constellation Shaping for OFDM-ISAC Systems: From Theoretical Bounds to Practical Implementation
## Benedikt Geiger, Fan Liu, Shihang Lu, Andrej Rode, Daniel Gil Gaviria, Charlotte Muth, and Laurent Schmalen

In OFDM-ISAC systems, the sensing performance (detection probability) depends on the kurtosis (fourth moment) of the transmit constellation. This repository provides a **MATLAB implementation** for the **numerical evaluation of upper and lower bounds on the achievable mutual information** of complex-valued signaling under **power and kurtosis constraints**.

The code computes:
- Lagrange multipliers of the maximum-entropy distribution subject to moment constraints,
- the resulting input and output entropies,
- and corresponding upper and lower mutual-information bounds as a function of the kurtosis constraint.
The implemented numerical procedures correspond to **(23)–(32)** in [1]

---

## Requirements

- MATLAB R2022b or later (tested with R2023b)
- Optimization Toolbox (for `fzero` and `fsolve`)

---

## Reference  

[1] B. Geiger, F. Liu, S. Lu, A. Rode, D. Gil Gaviria, C. Muth, and L. Schmalen, *“Constellation shaping for OFDM-ISAC systems: From theoretical bounds to practical implementation”*, submitted to IEEE TCOM, 2025. [arXiv:2509.04055](https://arxiv.org/abs/2509.04055)