# ISAC OFDM / FDM MATLAB Simulation

This repository contains MATLAB simulation codes for an
Integrated Sensing and Communication (ISAC) system based on
OFDM and FDM waveforms.

The project compares waveform-level BER performance between
ISAC-OFDM and ISAC-FDM under various SNR conditions.

---

## Folder Structure

- main/  
  Standalone waveform-level simulation scripts (OFDM / FDM)

- run/  
  Core simulation functions (called by analysis or demo scripts)

- functions/  
  Signal processing and channel model functions

- analysis/  
  Result comparison and visualization scripts

- outputs/  
  Generated simulation figures

---

## How to Run (Recommended)

1. Open MATLAB
2. Navigate to the `analysis` folder
3. Run:

```matlab
compare_ISAC_OFDM_FDM
```

This script automatically:
- Adds all required folders to the MATLAB path
- Runs both ISAC-OFDM and ISAC-FDM simulations
- Plots BER vs SNR comparison results

---

## Run Individual Simulations (Function Call)

### ISAC-OFDM
```matlab
gamma = 0.8;
SNR_dB = [-10 -5 0 5 10 15 20];

BER_OFDM = run_ISAC_OFDM(gamma, SNR_dB);
```

### ISAC-FDM
```matlab
alpha = 0.8;
SNR_dB = [-10 -5 0 5 10 15 20];

BER_FDM = run_ISAC_FDM(alpha, SNR_dB);
```

> Note:  
> `run_ISAC_OFDM.m` and `run_ISAC_FDM.m` are MATLAB functions and
> must be called with input arguments. They should not be executed
> directly using the Run button.

---

## Figure-to-Code Mapping

| Figure in Report | Description | MATLAB Script |
|------------------|-------------|---------------|
| Fig. 1 | Expected BER trends (OFDM vs FDM) | analysis/expected_ISAC_OFDM_FDM.m |
| Fig. 2 | ISAC-FDM BER under different α | main/main_ISAC_FDM_waveform.m |
| Fig. 3 | ISAC-OFDM BER under different γ | main/main_ISAC_OFDM_waveform.m |
| Fig. 4 | ISAC-OFDM vs ISAC-FDM comparison | analysis/compare_ISAC_OFDM_FDM.m |

All figures are saved in the `outputs/` folder.

---

## How to Reproduce Report Figures

- Fig. 1:
```matlab
cd analysis
expected_ISAC_OFDM_FDM
```

- Fig. 2:
```matlab
cd main
main_ISAC_FDM_waveform
```

- Fig. 3:
```matlab
cd main
main_ISAC_OFDM_waveform
```

- Fig. 4:
```matlab
cd analysis
compare_ISAC_OFDM_FDM
```

---

## Simulation Output

- BER vs SNR curves (log scale)
- Waveform-level performance comparison between ISAC-OFDM and ISAC-FDM

Random seeds are fixed to ensure reproducibility.

---

## Author

Department of Electronic Engineering  
National Ilan University
