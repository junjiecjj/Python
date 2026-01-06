
## 1. Overview

This archive contains the MATLAB simulation project used to generate Fig. 6 and Fig. 7 (and the related BER / hit-rate / CDF curves) in the paper.  
Its main purposes are:

- Build simulation scenarios for different system parameter settings (e.g., B = 160/320/640, T = 256/512, and 16QAM / 64QAM);
- Run Monte Carlo simulations of **bit error rate (BER)** and **target hit rate** under different signal-to-noise ratios (SNR);
- Save simulation results as `.mat` data files for plotting and repeated experiments;
- Provide a unified function library for target/channel/noise modeling, detection and parameter estimation;
- Provide plotting scripts to generate the final BER / SER / CDF / hit-rate figures used in the paper.

## 2. Directory Structure

```text
./
├── Data/                              # Simulation result data (.mat), e.g., BER / hit-rate curves for different B, T and QAM settings
│   └── BER_Hitrate/                   # Result files for different configurations (B_160_T_512, B_640_T_512_16QAM, etc.)
│
├── Function/                          # Core function library: target/trajectory generation, channel & noise modeling,
│                                      # DDMA sequence generation, detection and parameter estimation, etc.
│                                      # All main scripts use addpath('Function\') to call these utility functions.
│
├── Plot/                              # Plotting & post-processing scripts: read .mat files in Data/ and generate Fig. 6 / Fig. 7
│                                      # as well as CDF, BER, SER, hit-rate and other figures.
│                                      # Includes scripts such as CDF_Para.m, CDF_QAM.m, Plot_BER.m, Hitrate.m, and related .fig / .pdf files.
│
├── Main_BER_Hitrate_160_512.m         # Main simulation script: Monte Carlo BER and hit-rate simulation for B=160, T=512.
├── Main_BER_Hitrate_320_512.m         # Main simulation script: same as above but for B=320, T=512.
├── Main_BER_Hitrate_640_256.m         # Main simulation script: BER / hit-rate simulation for B=640, T=256.
├── Main_BER_Hitrate_640_512.m         # Main simulation script: BER / hit-rate simulation for B=640, T=512.
├── Main_BER_Hitrate_640_512_16QAM.m   # Main simulation script: B=640, T=512 with 16QAM modulation.
├── Main_BER_Hitrate_640_512_64QAM.m   # Main simulation script: B=640, T=512 with 64QAM modulation.
│                                      # In general, the Main_BER_Hitrate_* scripts:
│                                      #   1) Construct the corresponding ParaClass_* parameter object;
│                                      #   2) Set SNR sweep range and number of Monte Carlo trials;
│                                      #   3) Call functions in Function/ (e.g. TDDataGen) to generate received data and
│                                      #      count BER / hit rate;
│                                      #   4) Save the results into Data/BER_Hitrate/ for Plot/ scripts to use.
│
├── ParaClass_160_512.m                # System parameter class: all system and simulation parameters for B=160, T=512
├── ParaClass_320_512.m                # System parameter class: configuration for B=320, T=512
├── ParaClass_640_256.m                # System parameter class: configuration for B=640, T=256
├── ParaClass_640_512.m                # System parameter class: configuration for B=640, T=512
│                                      # ParaClass_* files provide a unified parameter interface to main scripts, including:
│                                      # SNR, noise PSD, transmit power, sampling rate, number of samples, frames/cycles,
│                                      # range/velocity resolution, target parameters, etc.
│
└── TDDataGen.m                        # Time-domain data generation function: uses ParaClass parameters and DDMA / QAM / IM data
                                       # to generate radar/communication receive-side time-domain data.
                                       # Called in Main_BER_Hitrate_* to build the received data cube with target echoes and noise.
