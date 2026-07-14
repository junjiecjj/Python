# RIS-Assisted FMCW Radar Target Detection in NLOS Scenario

王纪萱, 田团伟, 邓浩, 马锐. RIS辅助MIMO-FMCW雷达的非视距目标参数估计方法[J]. 现代雷达. DOI: [10.16592/j.cnki.1004-7859.2025203](https://dx.doi.org/10.16592/j.cnki.1004-7859.2025203)

## 1. Project Background

This project focuses on **RIS-assisted FMCW radar target detection in a non-line-of-sight (NLOS) scenario**.

In conventional FMCW radar systems, the radar usually relies on a direct line-of-sight propagation path to illuminate and detect targets. However, in practical environments such as urban roads, indoor corners, tunnels, and blocked sensing areas, the direct radar-target link may be obstructed by buildings, walls, vehicles, or other obstacles. In such cases, the target echo may become extremely weak or even unavailable.

To address this problem, this project introduces a **Reconfigurable Intelligent Surface (RIS)** to construct an additional controllable reflection path. By optimizing the RIS phase shifts, the indirect radar echo can be enhanced, thereby improving the target detection performance in the NLOS region.

The considered sensing link is:

```math
\text{Radar} \rightarrow \text{RIS} \rightarrow \text{Target} \rightarrow \text{RIS} \rightarrow \text{Radar}
```

The direct radar-target path is assumed to be blocked:

```math
\mathbf{h}_d = 0
```

Therefore, without RIS assistance, the radar receives only noise in the NLOS baseline.

---

## 2. System Scenario

The simulated scenario contains:

- A monostatic FMCW radar with transmit and receive antennas.
- A passive RIS with adjustable phase shifts.
- Multiple radar targets located in the NLOS region.
- A blockage between the radar and the targets.
- A controllable RIS-assisted indirect sensing path.

The equivalent RIS-assisted channel used in the current implementation is modeled as:

```math
\mathbf{H}_{\mathrm{eff}}
=
\mathbf{H}_{sr}^{H}
\mathbf{\Phi}
\mathbf{H}_{rd}
\mathbf{\Phi}^{H}
\mathbf{H}_{sr}
```

where:

| Symbol | Meaning |
|---|---|
| $\mathbf{H}_{sr}$ | Channel between radar/source and RIS |
| $\mathbf{H}_{rd}$ | RIS-domain equivalent target reflection channel |
| $\mathbf{\Phi}$ | RIS phase shift matrix |
| $\mathbf{H}_{\mathrm{eff}}$ | Equivalent RIS-assisted radar channel |

The RIS phase matrix is defined as:

```math
\mathbf{\Phi}=\mathrm{diag}(\mathbf{v})
```

with the unit-modulus constraint:

```math
|v_n|=1,\quad n=1,\ldots,N_{\mathrm{RIS}}
```

---

## 3. Main Objective

The main objective of this project is to verify whether RIS phase optimization can improve FMCW radar target detection performance in an NLOS scenario.

Specifically, this project aims to answer the following questions:

1. Can RIS phase optimization enhance the equivalent radar echo?
2. Can the enhanced echo be observed in the Range-Doppler map?
3. Can multiple targets be detected more reliably after RIS optimization?
4. How does detection probability vary with echo-domain SNR?
5. How do RIS hardware parameters, such as RIS element number and phase resolution, influence detection performance?

---

## 4. RIS Phase Optimization Method

After comparing several RIS phase optimization strategies, the current project adopts a fixed-grid coordinate phase search method.

The optimization target is the ZF-normalized SNR:

```math
\max_{\mathbf{v}} \quad SNR_{\mathrm{ZF}}(\mathbf{v})
```

subject to:

```math
|v_n|=1,\quad n=1,\ldots,N_{\mathrm{RIS}}
```

For discrete RIS phase control, each phase element is selected from a fixed grid:

```math
v_n \in \mathcal{G}
=
\left\{
e^{j2\pi l/L}
\mid
l=0,1,\ldots,L-1
\right\}
```

where $L$ is the number of discrete phase states.

The coordinate update is performed by searching the best phase value for each RIS element:

```math
v_n^{\star}
=
\arg\max_{v_n\in\mathcal{G}}
SNR_{\mathrm{ZF}}
(v_1,\ldots,v_n,\ldots,v_{N_{\mathrm{RIS}}})
```

This method is referred to as:

**Fixed-Grid Coordinate Phase Search for ZF-SNR Maximization**

or simply:

**Fixed-grid ZF-SNR RIS optimization**

This method is selected because it provides a clear performance gain with relatively low computational cost. More complex variants, such as coarse-to-fine search and condition-number-penalized search, showed only marginal improvement in the current experimental setting.

---

## 5. FMCW Echo and Range-Doppler Processing

The current FMCW radar signal model adopts a simplified dechirped beat signal form.

For the $q$-th target, the beat signal is modeled as:

```math
s_q[n,m]
=
A_q
\exp
\left[
j2\pi
\left(
f_{b,q} nT_s
+
f_{D,q} mT_c
\right)
\right]
```

where:

| Symbol | Meaning |
|---|---|
| $n$ | Fast-time sample index |
| $m$ | Chirp index |
| $T_s$ | ADC sampling interval |
| $T_c$ | Chirp repetition interval |
| $A_q$ | Complex amplitude of the target echo |
| $f_{b,q}$ | Beat frequency related to target range |
| $f_{D,q}$ | Doppler frequency related to target velocity |

The beat frequency is:

```math
f_{b,q}
=
\frac{2SR_q}{c}
```

where:

```math
S=\frac{B}{T_{\mathrm{chirp}}}
```

The Doppler frequency is:

```math
f_{D,q}
=
\frac{2v_q}{\lambda}
```

The received signal is:

```math
y[n,m]
=
\sum_{q=1}^{Q}
A_q
\exp
\left[
j2\pi
\left(
f_{b,q} nT_s
+
f_{D,q} mT_c
\right)
\right]
+
w[n,m]
```

where $w[n,m]$ is complex Gaussian noise.

Range-Doppler processing is implemented using two-dimensional FFT:

```math
\mathbf{Y}_{R}
=
\mathrm{FFT}_{n}
\{
\mathbf{Y}
\}
```

```math
\mathbf{Y}_{RD}
=
\mathrm{fftshift}
\left(
\mathrm{FFT}_{m}
\{
\mathbf{Y}_{R}
\}
\right)
```

The final RD map is displayed as:

```math
RD_{\mathrm{dB}}
=
20\log_{10}
\left(
|\mathbf{Y}_{RD}|+\epsilon
\right)
```

---

## 6. Experimental Design

The project currently contains several stages of experiments.

### 6.1 Stage 1: Channel and Model Validation

This stage verifies whether the RIS-assisted MIMO channel model is dimensionally consistent.

The equivalent channel is computed as:

```math
\mathbf{H}_{\mathrm{eff}}
=
\mathbf{H}_{sr}^{H}
\mathbf{\Phi}
\mathbf{H}_{rd}
\mathbf{\Phi}^{H}
\mathbf{H}_{sr}
```

ZF precoding is applied as:

```math
\mathbf{B}_{\mathrm{ZF}}
=
\sqrt{
\frac{P_t}
{
\|\mathbf{H}_{\mathrm{eff}}^{\dagger}\|_F^2
}
}
\mathbf{H}_{\mathrm{eff}}^{\dagger}
```

The ZF effective gain is:

```math
G_{\mathrm{ZF}}
=
\left\|
\mathbf{H}_{\mathrm{eff}}
\mathbf{B}_{\mathrm{ZF}}
\right\|_F^2
```

This stage confirms that the channel dimensions, ZF normalization, and SNR calculation are consistent.

---

### 6.2 Stage 2: RIS Phase Optimization Comparison

This stage compares several RIS phase strategies, including:

- Single random RIS phase.
- Best-of-random phase selection.
- Fixed-grid ZF-SNR optimization.
- Coarse-to-fine ZF-SNR optimization.
- Condition-number-penalized ZF-SNR optimization.

The results showed that fixed-grid ZF-SNR optimization provides the main performance gain, while more complex variants bring only limited additional improvement.

Therefore, the following experiments mainly use:

```math
\text{Fixed-grid ZF-SNR RIS optimization}
```

as the main RIS phase design method.

---

### 6.3 Stage 3: FMCW RD Map Verification

This stage connects the RIS-assisted equivalent gain to FMCW radar echo generation.

The RIS optimization gain is mapped to the target echo amplitude:

```math
A(\mathbf{v})
=
\sqrt{
G_{\mathrm{ZF}}(\mathbf{v})
}
\alpha
```

where $\alpha$ is the target reflection coefficient.

The goal of this stage is to verify:

```math
\text{RIS phase optimization}
\rightarrow
G_{\mathrm{ZF}} \text{ enhancement}
\rightarrow
RD \text{ peak enhancement}
```

The experiment first validates a single-target scenario and then extends to a four-target RD detection scenario.

---

### 6.4 Stage 4: Four-Target RD Detection

This stage evaluates the RD detection performance under a four-target NLOS setting.

The target set contains multiple targets with different ranges, velocities, and reflection coefficients:

```math
\{(R_q,v_q,\alpha_q)\}_{q=1}^{4}
```

The compared methods are:

| Method | Description |
|---|---|
| No RIS (NLOS) | Direct link is blocked; target echo amplitude is zero |
| Random RIS | RIS exists, but phases are randomly configured |
| Optimized RIS | RIS phases are optimized using fixed-grid ZF-SNR search |

The No RIS baseline is defined as:

```math
A_{\mathrm{NoRIS}}=0
```

```math
Y_{\mathrm{NoRIS}}[n,m]=w[n,m]
```

This represents a strict NLOS case where the radar cannot observe the targets without RIS assistance.

The four-target RD maps verify that the optimized RIS produces stronger target peaks than the random RIS case.

---

### 6.5 Stage 5: Pd-vs-SNR Detection Probability Experiment

This experiment studies the relationship between echo-domain SNR and detection probability.

The main horizontal axis is:

```math
\text{Echo SNR (dB)}
```

For each SNR point, the same noise power is used for No RIS, Random RIS, and Optimized RIS. Therefore, the performance difference is mainly caused by RIS-assisted gain and phase optimization.

The experiment compares:

| Method | Description |
|---|---|
| No RIS (NLOS) | Zero target echo, noise only |
| Random RIS | Random phase configuration |
| Optimized RIS | Fixed-grid ZF-SNR optimized phase configuration |

The detection probability is computed as:

```math
P_d
=
\frac{
N_{\mathrm{detected}}
}
{
N_{\mathrm{total}}
}
```

A full-map CA-CFAR detector is used before truth association.

---

## 7. CA-CFAR Detection and Truth Association

To make the detection experiment more realistic, the project uses full-map CA-CFAR detection.

For each cell under test, the CA-CFAR rule is:

```math
P_{\mathrm{CUT}}
>
\alpha_{\mathrm{CFAR}}
\hat{P}_{\mathrm{noise}}
```

where:

| Symbol | Meaning |
|---|---|
| $P_{\mathrm{CUT}}$ | Power of the cell under test |
| $\hat{P}_{\mathrm{noise}}$ | Estimated local noise power |
| $\alpha_{\mathrm{CFAR}}$ | CFAR threshold factor |

The CFAR threshold factor is:

```math
\alpha_{\mathrm{CFAR}}
=
N_{\mathrm{train}}
\left(
P_{\mathrm{fa}}^{-1/N_{\mathrm{train}}}
-
1
\right)
```

After CFAR detection, the detected points are associated with the four known ground-truth targets.

For each target, a detection is considered successful if a CFAR detection point lies within a small neighborhood of the true range and velocity:

```math
|\hat{R}-R_q|
\le
\Delta R_{\mathrm{tol}}
```

```math
|\hat{v}-v_q|
\le
\Delta v_{\mathrm{tol}}
```

If multiple CFAR detections fall into the same target neighborhood, the strongest RD peak is selected as the matched detection.

Unmatched CFAR detections are counted as false alarms.

This process allows the project to evaluate:

- Per-target detection probability.
- Average detection probability.
- Missed detections.
- False alarms per frame.
- Range and velocity estimation errors.

---

## 8. Stage 6: RIS Element Number and Phase Bit Sweep

This experiment is designed to evaluate the impact of RIS hardware parameters.

Two sub-experiments are considered.

### 8.1 RIS Element Number Sweep

The first sub-experiment varies the number of RIS elements:

```math
N_{\mathrm{RIS}}
\in
\{4,8,16,32,64\}
```

The phase resolution is fixed, for example:

```math
\text{phase bits}=4
```

or equivalently:

```math
L=16
```

The purpose is to study how RIS size influences:

- Equivalent ZF gain.
- RD peak enhancement.
- Detection probability.
- False alarms.
- Runtime.

Expected trend:

```math
N_{\mathrm{RIS}} \uparrow
\quad
\Rightarrow
\quad
\text{higher RIS-assisted gain and improved detection performance}
```

However, the gain may gradually saturate due to channel conditioning, finite target SNR, and limited phase resolution.

---

### 8.2 Phase Bit Sweep

The second sub-experiment varies the RIS phase resolution:

```math
b
\in
\{1,2,3,4,5\}
```

The number of phase states is:

```math
L=2^b
```

The purpose is to evaluate whether higher phase resolution improves detection performance.

Expected trend:

```math
b \uparrow
\quad
\Rightarrow
\quad
\text{better phase alignment and improved detection performance}
```

However, the performance improvement is expected to saturate after a few bits, indicating that a low-resolution RIS may be sufficient for practical deployment.

---

## 9. Current Experimental Conclusions

Based on the current implementation and experiments, the following conclusions can be drawn:

1. The RIS-assisted equivalent channel model is dimensionally consistent and can be connected to an FMCW RD processing chain.
2. Fixed-grid ZF-SNR phase optimization provides a clear gain over random RIS phases.
3. RIS optimization increases the RD target peak strength in both single-target and four-target scenarios.
4. In the NLOS baseline without RIS, the target echo is absent and the detection probability remains close to the false-alarm association level.
5. The optimized RIS achieves higher detection probability than random RIS under the same echo SNR condition.
6. RIS element number and phase resolution are important hardware parameters for evaluating the tradeoff between performance and complexity.

---

## 10. Limitations

The current project is still a staged simulation framework rather than a complete physical RIS-FMCW radar simulator.

The main limitations are:

1. The FMCW echo model currently uses a simplified dechirped beat signal model.
2. The RIS-assisted echo amplitude is mapped from the equivalent ZF gain rather than derived from a full geometric electromagnetic propagation model.
3. The current model does not yet explicitly simulate separate direct-path and RIS-path delays.
4. The current focus is Range-Doppler detection, while DOA estimation is not yet included.
5. Clutter, multipath interference, mutual coupling, and hardware phase errors are not yet fully modeled.
6. The current RIS optimization is based on fixed-grid search rather than a closed-form or advanced nonconvex optimization algorithm.

---

## 11. Future Work

Future development will focus on the following directions:

1. Introduce a more physical RIS-assisted FMCW propagation model.
2. Explicitly model direct path, RIS path, and their different delays.
3. Add clutter and multipath interference.
4. Extend the RD map to Range-Doppler-Angle processing.
5. Introduce DOA estimation algorithms such as FFT beamforming, CBF, and MUSIC.
6. Evaluate DOA estimation error under NLOS RIS-assisted sensing.
7. Study RIS deployment location, near-field effects, and target geometry.
8. Compare fixed-grid phase search with manifold optimization or other low-complexity RIS phase optimization methods.

---

## 12. Summary

This project builds a staged simulation framework for RIS-assisted FMCW radar target detection in an NLOS scenario.

The key idea is to use the RIS to provide a controllable indirect sensing path when the direct radar-target path is blocked. A fixed-grid coordinate phase search algorithm is used to optimize the RIS phase shifts for ZF-SNR enhancement. The enhanced equivalent gain is then connected to an FMCW beat signal model, followed by FFT-based Range-Doppler processing and CFAR-based target detection.

The current experiments show that optimized RIS phase control can enhance RD target peaks and improve detection probability compared with random RIS phase configuration under the same echo SNR condition.

The project is still under development, but it provides a useful foundation for further research on RIS-assisted FMCW radar sensing, NLOS target detection, and future DOA estimation.
