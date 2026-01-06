# README

## 1. Project Overview

This project is a MATLAB framework for joint communication–radar (or automotive radar) simulation and target tracking.  
The main functions include:

- Building system and scenario parameters (carrier frequency, bandwidth, array configuration, target trajectories)  
- Generating joint communication/radar transmit waveforms and time-domain echo data  
- Obtaining range–Doppler maps (RDM) via 2D FFT and detecting targets using CFAR  
- Estimating target range, velocity, angle, and orientation (heading)  
- Performing multi-target tracking and information fusion using Kalman filters (KF)  
- Saving simulation and tracking results as `.mat` files and visualizing/evaluating them with scripts in the `Plot` directory  

## 2. Directory Structure Overview

```text
.
├── Main_RadarTRx.m                % Main script for radar reception and tracking
├── Main_CommTRx.m                 % Main script for communication reception and tracking
├── ParaClass.m                    % System and scenario parameter class
├── TDDataGen.m                    % Time-domain data generation for radar/communication
├── Function/                      % Core algorithm function library
├── Data/                          % Simulation results and intermediate data
├── Fig/                           % Plotted figures
└── Plot/                          % Performance evaluation and visualization scripts
    ├── PerformanceEvaluation_Track_Radar.m
    └── PerformanceEvaluation_Track_Comm.m
