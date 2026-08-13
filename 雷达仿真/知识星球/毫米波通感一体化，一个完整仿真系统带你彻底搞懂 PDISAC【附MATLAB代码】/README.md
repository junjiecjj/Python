# PDISAC — MATLAB simulation and evaluation

This folder contains the MATLAB side of PDISAC: it simulates the multi-bit slot-partitioned PMCW waveform and the geometry-based channel, runs the conventional sensing/communication receivers, generates the RDPDNet training dataset, and evaluates the trained model. The `main1`…`main9b` scripts form the pipeline; they run in the MATLAB **base workspace** and share variables (each `mainN` assumes the earlier stages have populated the workspace, except the standalone drivers noted below).

## Requirements

- MATLAB R2022b or newer.
- **Phased Array System Toolbox** — `phased.Platform` (target/UE motion), `physconst`.
- **Database Toolbox** — `sqlite`, `sqlwrite` (writing the training `.db` for Python).
- Base MATLAB / Signal Processing — `fft`, `cart2sph`, `exportgraphics`, `readtable`/`writetable`.

## Configuration — `main1_matlab_config.m`

Single-file configuration, sourced by every other script (`main1_matlab_config;`). It defines the raw system/RF, waveform, scene, and CA-CFAR settings, derives all constants, and exposes profile handles (`dataset_*_cfg`, `*_statistics_cfg`, `PDISAC_cfg.inference`). Key blocks:

- **Waveform**: `chips` (`N_chip`), `blocks` (`N_prbs`), `symbols_per_block` (`N_bit^prbs`).
- **Scene**: ROI, moving targets, stationary scatterers, RCS ranges, speeds.
- **Dataset profiles**: `dataset_train` / `dataset_smoke` output paths (into `alg_pdisac/data/`).
- **RDPDNet inference bridge** (`PDISAC_cfg.inference.*`): `python_executable`, `checkpoint`, `device`, `batch_size`. **Set `python_executable` to your Python environment** before running `main9`/`main9b`.

## Pipeline: `main1` → `main9b`

| Script | Role |
|---|---|
| `main1_matlab_config.m` | Configuration (system/RF, waveform, scene, CFAR, dataset + inference bridge). Sourced by all others. |
| `main2_topology.m` | Builds the scene — UE, moving targets, stationary scatterers — and an optional 2-D plan view. |
| `main3_signal_channel_model.m` | Builds the transmit ISAC waveform and propagates it through the sensing and communication channels to form the received matrices. |
| `main4_sensing_process.m` | Matched filtering, RD-map formation, and static-clutter (zero-Doppler) removal; optional CA-CFAR + sinc-interpolation figures. |
| `main5_sensing_analysis.m` | Sensing RMSE vs SNR against the bias-adjusted CRLB (MLE + CA-CFAR pipeline). Writes `exported_statistics/statistics_sensing_biased_crlb.csv`. |
| `main5b_sensing_analysis.m` | Waveform-parameter sweeps — range/velocity RMSE vs `N_chip`/`N_prbs`. Writes `statistics_sensing_sweep_nchip.csv` / `_nprbs.csv`. |
| `main6_comm_process.m` | Communication decoding — delay estimation, pilot-slot channel estimation, BPSK demodulation. |
| `main7_comm_analysis.m` | BER and ensemble-average capacity analysis. Writes `exported_statistics/statistics_communication.csv`. |
| `main8_main.m` | End-to-end driver: aggregates the exported statistics into the paper's result figures (communication BER/capacity, sensing sweeps, channel distribution). |
| `main9_ml_sensing.m` | ML-sensing comparison — RDPDNet-denoised vs conventional RD maps (calls the Python inference bridge). |
| `main9b_ml_sensing.m` | Data-embedded vs data-free RD maps through RDPDNet, with an acceptance gate; produces the central sensing figures (RMSE vs SNR, bias significance) plus RD/AFM-mask visualizations. |

**Standalone entry points** (not part of the `main1`→`main9b` chain):

| Script | Role |
|---|---|
| `generate_radar_dataset.m` | Generates the RDPDNet training dataset `alg_pdisac/data/dataset_train.db` (or `dataset_smoke.db` when `PDISAC_SMOKE` is set). |
| `channel_com.m` | Channel LOS/NLOS distribution statistics → `exported_statistics/statistics_channel.mat` (consumed by `main8` Step 6b). |

**Helper functions** (called by the above): `func_ana_crlb_rv`, `func_bpsk_mod`/`func_bpsk_demod`, `func_ca_cfar_adaptive_threshold`, `func_generate_static_scatterers`, `func_get_array_response`, `func_get_x_delay`, `func_mle_estimation`, `func_sinc_interpolation`, `h_sen_tars`.

## Link to the Python code (`alg_pdisac/`)

The MATLAB and Python sides are coupled through files on disk, in two places.

**1. Dataset generation (MATLAB → Python).**
`generate_radar_dataset.m` writes the SQLite dataset that the Python trainer reads:

```matlab
generate_radar_dataset                       % → alg_pdisac/data/dataset_train.db
setenv('PDISAC_SMOKE','1'); generate_radar_dataset   % → alg_pdisac/data/dataset_smoke.db
```

`PDISAC_SMOKE` is the same environment variable the Python side uses, so it flips both dataset generation and training between the full and smoke configurations.

**2. Inference bridge (Python → MATLAB).**
During evaluation, `main9`/`main9b` call the trained RDPDNet through three thin MATLAB wrappers that launch the Python inference scripts (using `PDISAC_cfg.inference.python_executable` and `PDISAC_cfg.inference.checkpoint`):

| MATLAB wrapper | launches (in `alg_pdisac/scripts/`) | purpose |
|---|---|---|
| `inference_run_pdnet.m` | `inference_pdnet_mat.py` | denoise an RD map (or a batch) with RDPDNet |
| `inference_run_pdnet_mask.m` | `inference_pdnet_mask_mat.py` | rebuild the AFM adversarial mask |
| `inference_run_pdnet_distribution.m` | `distribution_pdnet_mat.py` | export the latent-distribution hierarchy for visualization |

The checkpoint the bridge reads (`PDISAC_cfg.inference.checkpoint`, default `alg_pdisac/exps/pdnet_afm/checkpoints/best_checkpoint.pth`) is the one written by `python -m scripts.train`.

## Typical order

```matlab
% 1) generate the dataset for Python training
generate_radar_dataset

% 2) train RDPDNet in Python (see alg_pdisac/README.md), then

% 3) run the analyses / evaluation
main5_sensing_analysis      % sensing RMSE vs CRLB
main5b_sensing_analysis     % sensing RMSE vs CRLB
main7_comm_analysis         % BER + capacity
main8_main                  % aggregate result figures
main9b_ml_sensing           % RDPDNet-denoised vs conventional sensing
```

## Outputs

- `exported_statistics/` — CSV/MAT statistics consumed by `main8_main.m`.
- `fig_exported/` — figures (`.fig`/`.png`) produced by the analysis scripts.
