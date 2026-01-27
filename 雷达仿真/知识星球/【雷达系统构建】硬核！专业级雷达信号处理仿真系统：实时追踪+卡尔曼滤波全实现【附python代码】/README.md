# Radar Signal Processing Simulator

A professional-grade radar simulation system demonstrating real-time signal processing, target detection, and multi-target tracking with Kalman filtering.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## Features

- **Real-time Radar Animation**: Professional military-style PPI (Plan Position Indicator) display with phosphor trail effect
- **Signal Processing Pipeline**: Noise filtering, radar range equation, SNR calculations
- **Target Detection**: Clustering, CFAR-inspired thresholding, automatic classification
- **Multi-Target Tracking**: Kalman filter implementation with track lifecycle management
- **Interactive Controls**: Start/stop system, scenario selection, real-time status monitoring
- **Multiple Scenarios**: Airport traffic, naval operations, weather tracking

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/radar-simulator.git
cd radar-simulator

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Run the interactive radar demo
python main.py

# Run component tests
python main.py --test

# Show system information
python main.py --info
```

## Usage

1. Launch the simulator with `python main.py`
2. Click **START SYSTEM** to begin radar operation
3. Watch targets appear on the radar scope as the sweep passes over them
4. Track information updates in real-time in the side panels
5. Click **STOP SYSTEM** to pause or **RESET SYSTEM** to reload the scenario

## Architecture

```
radar-simulator/
├── main.py                    # Entry point
├── complete_radar_system.py   # Main integrated demo
├── signal_processing.py       # Signal processing algorithms
├── target_detection.py        # Detection and classification
├── kalman_filter.py           # Kalman filter implementation
├── multi_target_tracker.py    # Multi-target tracking system
├── radar_data_generator.py    # Scenario and target generation
├── radar_display.py           # Basic display components
├── coordinate_utils.py        # Coordinate conversion utilities
├── signal_types.py            # Signal data types
├── radar_enums.py             # Shared enumerations
└── requirements.txt           # Python dependencies
```

### Component Overview

| Component | Description |
|-----------|-------------|
| **Signal Processing** | Implements radar range equation, noise filtering (moving average, exponential smoothing), SNR calculations |
| **Target Detection** | Clusters radar returns, applies threshold detection, classifies targets (aircraft/ship/weather) |
| **Kalman Filter** | 4-state filter (x, y, vx, vy) for position and velocity estimation with prediction and update cycles |
| **Multi-Target Tracker** | Manages multiple simultaneous tracks with association, confirmation, and termination logic |
| **Data Generator** | Creates realistic scenarios with physics-based target motion and environmental effects |

## Technical Details

### Radar Parameters
- **Max Range**: 150 km
- **Sweep Rate**: 30 RPM (configurable)
- **Detection Threshold**: Adaptive based on noise floor
- **Track Confirmation**: Requires consistent detections across sweeps

### Signal Processing Pipeline
1. Raw radar returns generated with realistic noise
2. Moving average and exponential smoothing filters applied
3. SNR-based threshold detection
4. Clustering of nearby returns
5. Classification based on Doppler, RCS, and signal characteristics

### Kalman Filter State Vector
```
State: [x, y, vx, vy]
- x, y: Position (km)
- vx, vy: Velocity (km/s)
```

## Screenshots

The simulator displays:
- Main radar scope with sweep animation
- System status indicators
- Tracked targets panel
- Performance metrics
- Alert messages

## Development

### Running Tests

```bash
python main.py --test
```

### Code Structure

The codebase follows clean architecture principles:
- Each module has a single responsibility
- Dependencies flow inward (display → processing → data)
- All modules are independently testable

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Radar fundamentals based on standard military radar systems
- Kalman filter implementation follows standard textbook formulations
- Display aesthetics inspired by real PPI radar displays

## Future Improvements

- [ ] Add more radar modes (track-while-scan, engagement)
- [ ] Implement JPDA for improved data association
- [ ] Add 3D tracking capability
- [ ] Export track data to standard formats
- [ ] Add configuration file support
