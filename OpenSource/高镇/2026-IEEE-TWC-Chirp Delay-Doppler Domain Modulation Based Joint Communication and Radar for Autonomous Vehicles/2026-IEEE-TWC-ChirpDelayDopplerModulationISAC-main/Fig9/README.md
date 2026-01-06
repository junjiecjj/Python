# Mutual Information Simulation for DDM/TDM Index-QAM

This repository contains **PyTorch** code for simulating the **mutual information / achievable rate** of different modulation schemes over random multipath channels. It focuses on comparing:

- Pure index modulation (IM)
- Pure QAM modulation
- Hybrid index + QAM modulation (Index-QAM)

under two different system structures:

- **DDM** – Doppler-division Multiplexing  
- **TDM** – Time-division Multiplexing

---

## Features

- Monte Carlo–based mutual information estimation (GPU-accelerated)
- Memory-efficient channel implementation (no need to store the full large matrix)
- Multiple QAM orders: 4 / 16 / 64
- Checkpointing support for long simulations (resume from where you stopped)
- Automatic export of results to `.csv` for plotting and further analysis

---

## Project Structure

```bash
.
├── Run.py                 # Top-level script: runs both DDM and TDM simulations
├── DDM/                   # DDM (Delay–Doppler) scenario: code and data
│   ├── Run.py             # Run all DDM simulations
│   ├── LowMem_Ckpt_Bar_IM_QAM_4.py       # 4-QAM, low-memory + checkpoint
│   ├── LowMem_Ckpt_Bar_IM_QAM_16.py      # 16-QAM
│   ├── LowMem_Ckpt_Bar_IM_QAM_64.py      # 64-QAM
│   ├── ReadCheckpoint.py                 # Load checkpoint and export CSV
│   ├── Data_256/          # Simulation results for N_f * N_c = 256
│   └── Data_512/          # Simulation results for N_f * N_c = 512
├── TDM/                   # TDM scenario: code and data (structure similar to DDM)
│   ├── Run.py
│   ├── LowMem_Ckpt_Bar_IM_QAM_4.py
│   ├── LowMem_Ckpt_Bar_IM_QAM_16.py
│   ├── LowMem_Ckpt_Bar_IM_QAM_64.py
│   ├── ReadCheckpoint.py
│   ├── Data_256/
│   └── Data_512/
└── Plot/                 # Visualization
```

The **DDM** and **TDM** directories share almost the same code structure, but use different system parameters (e.g., `N_f`, `N_c`) to represent different physical models / reference systems.

---

## Requirements

- Python >= 3.8
- PyTorch (CUDA recommended)
- NumPy
- Matplotlib
- Pandas

Example installation:

```bash
pip install torch numpy matplotlib pandas
```

(You may need to install the GPU version of PyTorch according to your CUDA setup.)

---

## Getting Started

### 1. Run all simulations

From the project root:

```bash
python Run.py
```

This will sequentially run:

- `DDM/Run.py`: all IM / QAM / Index-QAM simulations for the DDM scenario
- `TDM/Run.py`: all IM / QAM / Index-QAM simulations for the TDM scenario

During simulation, checkpoints are automatically saved and re-used, so interrupted runs can continue without starting from scratch.  
Results are stored in the corresponding `Data_256/` and `Data_512/` folders under `DDM/` and `TDM/`.

---

### 2. Run a specific scenario or modulation

Example: run only the DDM scenario with 16-QAM:

```bash
cd DDM
python LowMem_Ckpt_Bar_IM_QAM_16.py
```

Run the multi-channel mutual information demo (plots only, no CSV export):

```bash
cd DDM
python AchievableRate_MultiChannel.py
```

Run the single-channel mutual information demo:

```bash
cd DDM
python AchievableRate_OneChannel.py
```

TDM scripts can be run in the same way inside the `TDM/` folder.

---

### 3. Reading results and exporting CSV

Use the provided helpers to convert checkpoint files (`.pt`) into `.csv`:

```bash
cd DDM
python ReadCheckpoint.py
```

Inside `ReadCheckpoint.py`, set:

- `ckpt_file`: path to the `.pt` checkpoint you want to read
- `save_file`: output `.csv` path

The script will:

1. Load the tensor `mi_all` (mutual information vs SNR for multiple realizations)
2. Attach column names such as `SNR=0dB`, `SNR=5dB`, ...
3. Save the result as a CSV file for easy plotting (e.g., in Python, MATLAB, or Excel)

The `TDM/ReadCheckpoint.py` script works in exactly the same way for TDM results.

---

## Notes

- DDM vs TDM:
  - **DDM** typically uses a 2D delay–Doppler grid (e.g., `N_f=512`, `N_c=32`) and applies index/QAM modulation across both dimensions.
  - **TDM** uses different parameter choices (e.g., `N_f=512`, `N_c=1` in some cases), representing a more traditional time-domain / QAM-like reference system.
- Mutual information is estimated via Monte Carlo sampling with GPU acceleration and a memory-efficient channel representation, so it can handle large-dimensional Index-QAM scenarios.
