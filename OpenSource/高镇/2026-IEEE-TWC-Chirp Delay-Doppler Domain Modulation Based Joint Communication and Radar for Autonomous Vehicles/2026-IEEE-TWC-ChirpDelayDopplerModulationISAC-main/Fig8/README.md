This archive contains code and simulation results for a **single-tone + QPSK** system under:
- **No prediction error** (`NoPredError`)
- **With prediction error / estimation error** (`PredError`)

For each setting, it includes:
- Python / MATLAB simulation and plotting scripts (FRaC scheme vs. baseline “Mine” scheme)
- JSON result files (BER / SER curves for different detection modes / cases)
- Plotted figures (BER / SER performance)

---

## File Structure

```text
.                               # Root directory of the archive
├── NoPredError/                # Experiments with perfect frequency knowledge (no prediction error)
│   ├── BER_FRaC.png            # BER curve for the FRaC scheme (no prediction error)
│   ├── BER_Mine.png            # BER curve for the Mine (single-tone + QPSK) baseline (no prediction error)
│   ├── fig/                    # MATLAB figure exports for SER comparison
│   │   ├── SER_Cmb.fig         # Combined SER figure (FRaC vs. Mine)
│   │   └── SER_Cmb.pdf         # PDF export of the combined SER figure
│   ├── FRaC.json               # FRaC results (modes: 'spatial', 'qpsk', 'all') with no prediction error
│   ├── FRaC.py                 # FRaC Monte-Carlo simulation script (no prediction error)
│   ├── Mine.json               # Mine results with three scenarios: joint tone+QPSK, tone-only, QPSK-only
│   ├── Mine.py                 # Mine Monte-Carlo simulation script (no prediction error, defines CASE 1/2/3)
│   ├── Plot_FRaC.m             # MATLAB plotting script for FRaC results
│   ├── Plot_FRaC.py            # Python plotting script for FRaC results
│   ├── Plot_Mine.m             # MATLAB plotting script for Mine results
│   ├── Plot_Mine.py            # Python plotting script for Mine results
│   ├── Plot_Simult.m           # MATLAB script: simultaneous comparison (FRaC vs. Mine, BER/SER)
│   ├── SER_FRaC.png            # SER curve for the FRaC scheme (no prediction error)
│   └── SER_Mine.png            # SER curve for the Mine baseline (no prediction error)
└── PredError/                  # Experiments with frequency-estimation error / prediction error
    ├── Fig/                    # Figures and plotting scripts for the prediction-error case
    │   ├── FRaC.fig            # MATLAB figure for FRaC with prediction error
    │   ├── Mine.fig            # MATLAB figure for Mine with prediction error
    │   ├── Plot_FRaC.m         # MATLAB plotting script for FRaC (prediction-error case)
    │   └── Plot_Mine.m         # MATLAB plotting script for Mine (prediction-error case)
    ├── FRaC.py                 # FRaC Monte-Carlo simulation with prediction error (modes: spatial / qpsk / all)
    ├── FRaC_0d0_1.json         # FRaC BER/SER curves, prediction-error setting “0d0” (spatial-only, QPSK-only, joint)
    ├── FRaC_0d1_1.json         # FRaC BER/SER curves, prediction-error setting “0d1” (same three modes)
    ├── FRaC_0d2_1.json         # FRaC BER/SER curves, prediction-error setting “0d2” (same three modes)
    ├── Mine.py                 # Mine Monte-Carlo simulation with prediction error (defines CASE 1/2/3)
    ├── Mine_0d0_1.json         # Mine results for setting “0d0”: joint tone+QPSK, tone-only, QPSK-only
    ├── Mine_0d1_1.json         # Mine results for setting “0d1”: joint tone+QPSK, tone-only, QPSK-only
    ├── Mine_0d2_1.json         # Mine results for setting “0d2”: joint tone+QPSK, tone-only, QPSK-only
    └── Mine_0d2_2.json         # Mine results for another “0d2” run: joint tone+QPSK, tone-only, QPSK-only (repeat/variant)
