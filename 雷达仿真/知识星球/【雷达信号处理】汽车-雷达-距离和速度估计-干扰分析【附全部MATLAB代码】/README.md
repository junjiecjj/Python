# Automotive-Radar-Range and Velocity Estimations-Interference-Analysis

This repository hosts a comprehensive analysis project focusing on the impact of Frequency-Modulated Continuous-Wave (FMCW) radar interference on target detection within automotive applications. The goal is to explore how different levels and types of interference can affect the accuracy of radar-based estimations of range and velocity—critical factors in the development and performance of automotive safety and navigation systems.

## Key Objectives

- **Impact Analysis**: Investigate how FMCW radar interference influences target detection accuracy, specifically looking at how it affects the range and velocity measurements of nearby objects.
- **Data-Driven Insights**: Utilize collected data to analyze and visualize how interference disrupts the radar's ability to accurately detect and track targets.
- **Filter Implementation**: Test and demonstrate the effectiveness of various filtering techniques designed to mitigate the negative effects of radar interference, enhancing the radar's performance and reliability.

## Data Description

The dataset includes parameters from both a victim and an interferer radar system:

- **Victim Radar Parameters**:
  - EIRP: -5 dBW
  - Antenna Gain: 10 dBi
  - Start Frequency: 76.5 GHz
  - Bandwidth: 500 MHz
  - Sweep Time: 50 µs
  - Pulse Repetition Interval: 60 µs
  - Coherent Processing Interval: 10 ms
  - Low-Pass Filter Cut-off Frequency: 3.5 MHz
  - Signal Modulation: FMCW
  - Antenna Configuration: Single node (Initial), TDMA-MIMO (Advanced Stage)
  - Velocity: (0, 0) m/s

- **Interferer Radar Parameters**:
  - EIRP: -5 dBW
  - Start Frequency: 76.45 GHz
  - Bandwidth: 900 MHz
  - Sweep Time: 25 µs
  - Pulse Repetition Interval: 30 µs
  - Signal Modulation: FMCW

- **Targets**:
  - Target 1: Dimensions (4 m x 1.9 m x 1.5 m), Velocity (10, 0) m/s, Position (40, 3.6) m
  - Target 2: Dimensions (4.5 m x 1.9 m x 1.5 m), Velocity (-5, 0) m/s, Position (20, 7.2) m

## MATLAB Code

The `FMCW_Radar.m` MATLAB script processes the radar data to analyze the impact of interference and evaluates different filtering techniques. The code calculates radar signal properties, processes signal interference, and generates outputs that demonstrate the effects of various filters.

## Visual Outputs

The MATLAB script produces several plots that illustrate the radar signal characteristics and the effectiveness of different filtering approaches. these images are presented in the file named images.

