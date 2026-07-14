"""
Global configuration parameters for 3D MIMO Beamforming System
Centralized parameter control.

"""

import numpy as np

# Physical constants
C = 3e8
FC = 77e9
LAMBDA = C / FC
ELEMENT_SPACING = LAMBDA / 2

# Array configuration
NX = 4
NY = 4
NUM_ELEMENTS = NX * NY

# Simulation parameters
NUM_SNAPSHOTS = 300
SNR_DB = 20
NUM_TARGETS = 2

# Target definitions (Azimuth, Elevation in degrees)
TARGETS = [
    (20, 15),
    (-35, 10)
]

# Scan grid
AZIMUTH_SCAN = np.linspace(-90, 90, 181)
ELEVATION_SCAN = np.linspace(-30, 30, 121)