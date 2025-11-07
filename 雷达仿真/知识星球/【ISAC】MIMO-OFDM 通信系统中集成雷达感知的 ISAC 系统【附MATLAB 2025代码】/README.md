# MIMO OFDM ISAC Simulation

This repository contains MATLAB code for simulating a MIMO-OFDM Integrated Sensing and Communication (ISAC) system. The simulation creates and evaluates data and radar-like sensing results for different system configurations. The code was taken from a [Matlab example](https://es.mathworks.com/help/phased/ug/integrated-sensing-and-communication-2-communication-centric-approach-using-mimo-ofdm.html), and has been refactored and extended with features such as CSV-based configuration for antennas and scenarios, modular task scripts, support for multiple system variations (antennas, subcarriers, SCS, datastreams), and automated result output grouping.

## Contents

- `main.m` — Main entry point to run the simulation.
- `configs/` — CSV configuration files for antennas, subcarriers, SCS, datastreams, targets, and default parameters.
- `functions/` — Helper functions, task scripts and utilities used by the simulation.
  - `./tasks/` — High-level tasks such as `configureSystem.m`, `configureScenario.m`, `transmitDataFrames.m`, and `processRadarData.m`.
  - `./helpers/` — Smaller helper utilities used across the project.
  - `./utils/.`- Utility package.
- `results/` — Output images and text results grouped by configuration.

## Prerequisites

- MATLAB (recommended R2024a or newer).
- Recommended MATLAB toolboxes (depending on which parts you use):
  - Communications Toolbox
  - Phased Array System Toolbox

If you don't have the exact toolboxes, the core scripts may still run but some functions might be unavailable. Edit the code or remove toolbox-specific calls if necessary.

## Quick start

1. Open MATLAB and add the repository directory to the MATLAB path, or change the current folder to the repository root.

2. Run the main simulation from the MATLAB prompt:

```matlab
run('main.m')
% or simply
main
```

## Configuration

If you want to change the configuration setup, change configuration file in main and then run the simulation:

```matlab
%% Configuration
params = 'configs/config_antennas.csv';
```

Edit CSV files under `configs/` to change scenarios and parameter sets. The scripts read these files at runtime (see `read_params.m`) and apply the requested configuration.

## Outputs

Simulation outputs (plots and plain text results) are written into the `results/` directory. Subfolders are organized by configuration type (e.g., `antennas/`, `datastreams/`, `scs/`, `default/`).