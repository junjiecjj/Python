# Radar Pulse Simulation in MATLAB 🚀

Welcome to the **Radar Pulse Simulation** project! This repository hosts a comprehensive simulation of a pulsed radar system, designed as part of a Master's project in Telecommunications at UAM. The simulation includes parameter calculations, fluctuating moving targets, and PPI/A-Scope visualization.

[![Download Releases](https://img.shields.io/badge/Download_Releases-Click_here-brightgreen)](https://github.com/SebaPythonGPT/radar-pulse-simulation-matlab/releases)

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Simulation Details](#simulation-details)
   - [Parameters](#parameters)
   - [Target Detection](#target-detection)
7. [Visualization](#visualization)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

## Introduction

Radar technology plays a crucial role in various fields, including telecommunications, aviation, and automotive safety. This project simulates a pulsed radar system, focusing on key aspects such as noise modeling, target detection, and signal processing. The simulation provides insights into radar operations and enhances understanding of radar principles.

## Features

- **Pulsed Radar Simulation**: Simulates the behavior of a pulsed radar system.
- **Moving Targets**: Models fluctuating targets to evaluate detection capabilities.
- **Parameter Calculation**: Calculates essential radar parameters.
- **PPI and A-Scope Visualization**: Displays radar data in intuitive formats.
- **Signal Processing**: Implements techniques for effective signal analysis.
- **Noise Modeling**: Accounts for environmental noise in simulations.

## Getting Started

To get started with the Radar Pulse Simulation project, follow these steps:

### Prerequisites

Ensure you have the following software installed:

- MATLAB (R2018a or later)
- Signal Processing Toolbox
- Communications Toolbox

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SebaPythonGPT/radar-pulse-simulation-matlab.git
   ```

2. Navigate to the project directory:

   ```bash
   cd radar-pulse-simulation-matlab
   ```

3. Download the latest release from the [Releases](https://github.com/SebaPythonGPT/radar-pulse-simulation-matlab/releases) section. Extract the files and place them in the project directory.

## Usage

To run the simulation:

1. Open MATLAB.
2. Navigate to the project directory.
3. Execute the main script:

   ```matlab
   run_simulation.m
   ```

4. Follow the prompts to configure simulation parameters.

## Project Structure

```
radar-pulse-simulation-matlab/
├── data/
│   └── target_data.mat
├── src/
│   ├── run_simulation.m
│   ├── calculate_parameters.m
│   ├── target_detection.m
│   └── visualize_results.m
├── results/
│   └── output_data.mat
└── README.md
```

- **data/**: Contains sample data for target simulation.
- **src/**: Source code files for running the simulation.
- **results/**: Stores output data from the simulation.

## Simulation Details

### Parameters

The radar simulation requires various parameters, including:

- **Pulse Width**: Duration of each pulse.
- **Pulse Repetition Frequency (PRF)**: Rate at which pulses are transmitted.
- **Operating Frequency**: Frequency of the radar signal.
- **Detection Threshold**: Minimum signal level for target detection.

Adjust these parameters in the `run_simulation.m` file to tailor the simulation to your needs.

### Target Detection

The project implements several algorithms for target detection, including:

- **Constant False Alarm Rate (CFAR)**: Adjusts the detection threshold based on noise levels.
- **Swerling Models**: Models target behavior under different conditions.

Experiment with different algorithms to see how they affect detection performance.

## Visualization

The simulation includes visualization tools for better understanding of radar data:

- **PPI Display**: Presents the radar data in a polar plot.
- **A-Scope**: Shows the signal strength over time for a specific range.

To visualize the results, call the `visualize_results.m` function after running the simulation.

## Contributing

We welcome contributions to improve this project. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push to your branch and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the faculty and peers at UAM for their support during this project.
- Inspiration drawn from existing radar systems and simulation frameworks.

For more information, visit the [Releases](https://github.com/SebaPythonGPT/radar-pulse-simulation-matlab/releases) section to download the latest version of the simulation files. 

Feel free to explore, modify, and enhance the simulation as per your requirements!